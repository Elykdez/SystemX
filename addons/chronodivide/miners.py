"""Brief early-rush support from one separately verified War Miner."""

from dataclasses import dataclass

from .executor import Step
from .perception.enemy_threats import deployed_gi_targets, standing_gi_targets
from .perception.resources import gem_field, ore_supply, ore_target
from .perception.unit_labels import read_label
from .perception.unitboxes import our_units
from .tasks import Act, Look, Poll


@dataclass
class MinerGuard:
    grouped: bool = False
    attempts: int = 0
    engaged: bool = False
    returning: bool = False
    started: float = 0
    checked: float = -1e9
    last_used: float = -1e9
    home_ore_seen: bool | None = None
    home_ore_chips: int = 0
    home_ore_at: float = -1e9
    home_ore_thin_since: float | None = None
    home_ore_target: tuple | None = None
    funding_at: float | None = None
    income_at: float = -1e9
    income_credits: int | None = None
    income_progress: tuple = ()
    field: tuple | None = None
    field_at: float = -1e9
    field_chips: int = 0
    field_resource: str = 'gems'
    resource_phase: str = 'idle'
    resource_at: float = -1e9
    resource_last: float = -1e9
    resource_credits: int | None = None
    survey_at: float = -1e9
    survey_direction: int = 0


def observe_home(ctx, state, plan, now):
    if (ctx.camera != 'home' or ctx.last_canvas is None or
            now - state.home_ore_at < plan['home_check_every']):
        return
    region = ctx.layout.tactical
    state.home_ore_chips = ore_supply(ctx.last_canvas, ctx.home, plan['home_reach'],
                                     (region.x, region.y, region.w, region.h))
    state.home_ore_seen = state.home_ore_chips >= plan['home_min_chips']
    state.home_ore_target = ore_target(ctx.last_canvas, ctx.home, plan['home_reach'],
                                      (region.x, region.y, region.w, region.h))
    state.home_ore_at = now
    if state.home_ore_seen:
        state.home_ore_thin_since = None
    elif state.home_ore_thin_since is None:
        state.home_ore_thin_since = now
    observe_field(ctx, state, plan, now)


def find_field(canvas, region, plan, near):
    gems = gem_field(canvas, region)
    if gems is not None:
        return (*gems, 'gems')
    if plan.get('survey_ore'):
        point = ore_target(canvas, near, plan['home_reach'], region)
        chips = ore_supply(canvas, near, plan['home_reach'], region)
        if point is not None and chips >= plan['home_min_chips']:
            return point, chips, 'ore'
    return None


def observe_field(ctx, state, plan, now):
    radar = ctx.world.radar.value
    if ctx.last_canvas is None or radar is None or not radar.online or radar.view is None:
        return
    box = ctx.layout.tactical
    found = find_field(ctx.last_canvas, (box.x, box.y, box.w, box.h), plan, ctx.home)
    if found is None:
        return
    point, chips, resource = found
    x0, y0, x1, y1 = radar.view
    rx = x0 + round((point[0] - box.x) / box.w * (x1 - x0 + 1))
    ry = y0 + round((point[1] - box.y) / box.h * (y1 - y0 + 1))
    target = radar.inside((rx, ry))
    home = ctx.radar_home
    if (home is None or abs(target[0] - home[0]) + abs(target[1] - home[1]) >
            plan['max_radar_distance'] or radar.explored_near(target, radius=2) < 0.6):
        return
    if plan.get('survey_ore') and radar.enemy_near(target, plan['survey_enemy_radius']):
        return
    state.field, state.field_at, state.field_chips = target, now, chips
    state.field_resource = resource
    ctx.world.note('resource_field', resource=resource, radar=list(target), chips=chips)


def home_thin(state, plan, now):
    return (state.home_ore_thin_since is not None and
            now - state.home_ore_thin_since >= plan['home_thin_seconds'] and
            now - state.home_ore_at <= plan['home_fresh_within'])


def field_fresh(state, plan, now):
    return state.field is not None and now - state.field_at <= plan['field_fresh_within']


def income_stalled(state, plan, now):
    return (state.funding_at is not None and now - state.income_at <= plan['home_fresh_within'] and
            now - state.funding_at >= plan['income_stall_seconds'])


def home_ore_ready(state, plan, now):
    return (state.home_ore_seen and state.home_ore_target is not None and
            now - state.home_ore_at <= plan['home_fresh_within'])


def survey_point(army):
    state, plan = army.miners, army.book['resources']
    radar, home = army.ctx.world.radar.value, army.ctx.radar_home
    if radar is None or not radar.online or radar.view is None or home is None:
        return None
    points = []
    for dx, dy in plan['survey_offsets']:
        point = radar.inside((home[0] + dx, home[1] + dy))
        if (point != home and point not in points and
                radar.explored_near(point, radius=2) >= plan['survey_min_explored'] and
                not radar.enemy_near(point, plan['survey_enemy_radius'])):
            points.append(point)
    return points[state.survey_direction % len(points)] if points else None


def survey_due(army, now):
    state, plan = army.miners, army.book['resources']
    credits = army.ctx.world.spendable()
    return (plan['enabled'] and state.grouped and not state.engaged and
            state.resource_phase == 'idle' and not army.campaign.active and
            army.intent is not None and army.intent.resources == 'survey_gems' and
            credits is not None and credits < plan['low_credits_below'] and
            army.threat_age(now) >= plan['quiet_seconds'] and
            now - state.survey_at >= plan['survey_every'] and
            home_thin(state, plan, now) and not field_fresh(state, plan, now) and
            survey_point(army) is not None)


def resource_due(army, now):
    state, plan = army.miners, army.book['resources']
    if (not plan['enabled'] or state.engaged or not state.grouped or army.campaign.active or
            army.threat_age(now) < plan['quiet_seconds']):
        return False
    if state.resource_phase == 'travel':
        return now - state.resource_at >= plan['travel_wait']
    choice = army.intent.resources if army.intent is not None else None
    target_ready = ((choice == 'harvest_gems' and home_thin(state, plan, now) and
                     field_fresh(state, plan, now)) or
                    (choice == 'harvest_home_ore' and income_stalled(state, plan, now) and
                     home_ore_ready(state, plan, now)))
    return (state.resource_phase == 'idle' and target_ready and
            army.ctx.world.spendable() is not None and
            army.ctx.world.spendable() < plan['low_credits_below'] and
            now - state.resource_last >= plan['retry_after'])


def monitor_income(ctx, state, plan, now):
    credits = ctx.world.credits.value
    if credits is not None:
        if state.income_credits is None or credits > state.income_credits:
            state.funding_at = now
        progress = dict(state.income_progress)
        for name, lane in ctx.world.lanes.items():
            previous = progress.get(name)
            current = (lane.item, lane.progress, lane.status)
            if (previous is None or previous[0] != lane.item or
                    (lane.progress is not None and (previous[1] is None or
                     lane.progress < previous[1] - 0.03))):
                progress[name] = current
                continue
            advanced = (lane.progress is not None and previous[1] is not None and
                        lane.progress - previous[1] >= 0.03)
            if advanced or (lane.status == 'ready' and previous[2] == 'building'):
                # Spending can consume deliveries before the next credit sample.
                state.funding_at = now
                progress[name] = current
        state.income_progress = tuple(progress.items())
        state.income_credits, state.income_at = credits, now
    if state.resource_phase != 'ordered':
        return
    if (credits is not None and state.resource_credits is not None and
            credits > state.resource_credits):
        ctx.world.note('miner_resource', status='cash rose after resource order; source unverified',
                       before=state.resource_credits, after=credits)
        state.resource_phase = 'idle'
    elif now - state.resource_at >= plan['income_watch']:
        ctx.world.note('miner_resource', status='no cash rise observed after resource order')
        state.resource_phase = 'idle'


def focus(ctx, team):
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key(team), Step.wait(80),
               Step.key(team), Step.wait(180), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'away'
    return one_miner((yield Look(lambda s, v: selected(ctx))))


def resource_order(ctx, state, plan, clock, resource='gems'):
    now = clock()
    unit = yield from focus(ctx, plan['team'])
    if unit is None or unit.health is None or unit.health < plan['miner_min_health']:
        state.grouped = False if unit is None else state.grouped
        state.resource_phase, state.resource_last = 'idle', now
        ctx.world.note('miner_resource', status='saved miner missing or unhealthy')
    elif resource == 'ore':
        yield Act([Step.key('h'), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.25)
        ctx.camera = 'home'
        box = ctx.layout.tactical
        point = yield Look(lambda s, v: ore_target(ctx.last_canvas, ctx.home, plan['home_reach'],
                           (box.x, box.y, box.w, box.h)))
        state.resource_last = clock()
        if point is not None:
            state.resource_phase, state.resource_at = 'ordered', clock()
            state.resource_credits = ctx.world.credits.value
            yield Act([Step.click(point), Step.move(ctx.park)], commands=1)
            ctx.world.note('miner_resource', status='home ore harvest order sent', target=list(point))
        else:
            ctx.world.note('miner_resource', status='home ore no longer verified')
    elif state.resource_phase == 'idle':
        if field_fresh(state, plan, now):
            state.resource_phase, state.resource_at = 'travel', clock()
            yield Act([Step.click(state.field), Step.move(ctx.park)], commands=1)
            ctx.world.note('miner_resource', status='move order sent', radar=list(state.field))
    elif state.resource_phase == 'travel':
        box = ctx.layout.tactical
        field = yield Look(lambda s, v: find_field(ctx.last_canvas,
                            (box.x, box.y, box.w, box.h), plan, unit.center))
        if field and field[2] != state.field_resource:
            field = None
        near = field and sum(abs(a - b) for a, b in zip(field[0], unit.center)) <= plan['gem_reach']
        if near:
            point, chips, resource = field
            state.resource_phase = 'ordered'
            state.resource_at = state.resource_last = clock()
            state.resource_credits = ctx.world.credits.value
            yield Act([Step.click(point), Step.move(ctx.park)], commands=1)
            label = 'gem' if resource == 'gems' else resource
            ctx.world.note('miner_resource', status=f'{label} harvest order sent',
                           target=list(point), chips=chips)
        elif clock() - state.resource_at >= plan['travel_timeout']:
            state.resource_phase, state.resource_last = 'idle', clock()
            ctx.world.note('miner_resource', status='travel timed out without fresh gems')
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.2)
    ctx.camera = 'home'


def can_group(world, state, plan):
    # Before the first factory, the deployed MCV's refinery supplies the sole vehicle.
    return (plan['enabled'] and not state.grouped and state.attempts < plan['group_attempts'] and
            world.mcv_deployed.value and world.structures.get('ore_refinery') == 1 and
            not world.structures.get('war_factory') and
            not any(n for item, n in world.units_built.items() if world.item(item)['lane'] == 'vehicles'))


def can_recover_group(army, now):
    state, plan = army.miners, army.book['defense']['emergency_miners']
    resources = army.book['resources']
    return (plan['enabled'] and not state.grouped and
            state.attempts < plan['group_attempts'] and not army.campaign.active and
            army.threat_age(now) >= resources['quiet_seconds'] and
            now - state.resource_last >= resources['retry_after'] and
            income_stalled(state, resources, now) and
            army.ctx.world.spendable() is not None and
            army.ctx.world.spendable() < resources['low_credits_below'])


def recover_group(ctx, state, plan):
    state.attempts += 1
    state.resource_last = ctx.world.clock()
    funding_at = state.funding_at
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h')], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    expected = 'war_miner' if plan.get('armed', True) else 'chrono_miner'
    # N cycles actual units; only a fresh native tooltip can identify a late-game miner.
    ctx.camera = 'away'
    for _ in range(32):
        yield Act([Step.key('n'), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.15)
        unit = one_miner((yield Look(lambda s, v: selected(ctx))))
        if state.funding_at != funding_at:
            break                       # income resumed; stop the obsolete search
        if unit is None or unit.health is None or unit.health < plan['min_health']:
            continue
        point = (unit.bar[0] + unit.bar[2] // 2, unit.bar[1] + 25)
        yield Act([Step.move(ctx.clear_point), Step.move(point), Step.wait(800)], commands=0)
        name = yield Look(lambda s, v: read_label(ctx.last_canvas, point))
        if name != expected:
            continue
        yield Act([Step.chord('ctrl', plan['team']), Step.move(ctx.park)], commands=0)
        saved = one_miner((yield from select(ctx, [plan['team']])))
        state.grouped = saved is not None
        break
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    ctx.world.note('miner_group', verified=state.grouped, team=plan['team'], source='native tooltip recovery')


def selected(ctx):
    return our_units(ctx.last_canvas, compressed=ctx.compressed_frames) if ctx.last_canvas is not None else []


def one_miner(units):
    return units[0] if len(units) == 1 and units[0].kind == 'vehicle' else None


def select(ctx, keys):
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    yield Act([Step.key(k) for k in keys] + [Step.wait(150), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.12)
    return (yield Look(lambda s, v: selected(ctx)))


def group(ctx, state, plan):
    state.attempts += 1
    unit = one_miner((yield from select(ctx, ['n'])))
    if unit is not None:
        yield Act([Step.chord('ctrl', plan['team'])], commands=0)
        saved = one_miner((yield from select(ctx, [plan['team']])))
        state.grouped = saved is not None
    yield Act([Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=0)
    ctx.world.note('miner_group', verified=state.grouped, team=plan['team'])


def ground_target(ctx, center, reach):
    if ctx.last_canvas is None or ctx.camera != 'home':
        return None
    box = ctx.layout.tactical
    region = (box.x, box.y, box.w, box.h)
    points = (standing_gi_targets(ctx.last_canvas, region, compressed=ctx.compressed_frames) +
              deployed_gi_targets(ctx.last_canvas, region, compressed=ctx.compressed_frames))
    points = [p for p in points if abs(p[0] - center[0]) + abs(p[1] - center[1]) <= reach]
    return min(points, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1])) if points else None


def due(army, now):
    state, plan = army.miners, army.book['defense']['emergency_miners']
    if state.engaged:
        return now - state.checked >= plan['check_every']
    census_stale = now - army.census_at > army.book['army']['census']['fresh_within']
    return (plan['enabled'] and plan.get('armed', True) and state.grouped and now - state.last_used >= plan['cooldown'] and
            (census_stale or army.census_healthy_vehicles <= plan['max_defenders']) and
            ground_target(army.ctx, army.ctx.home, plan['reach']) is not None)


def support(ctx, state, plan, clock):
    state.checked = clock()
    if not state.engaged:
        state.last_used = clock()
        defenders = yield from select(ctx, ['p'])
        healthy = sum(u.kind == 'vehicle' and u.health is not None and
                      u.health >= plan['defender_min_health'] for u in defenders)
        if healthy > plan['max_defenders']:
            yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
            return
    unit = one_miner((yield from select(ctx, [plan['team']])))
    if unit is None:
        state.grouped = state.engaged = state.returning = False
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        ctx.world.note('miner_support', status='team missing or selection uncertain')
        return
    target = yield Look(lambda s, v: ground_target(ctx, unit.center, plan['reach']))
    ore = yield Look(lambda s, v: ore_target(ctx.last_canvas, unit.center, plan['ore_reach']))
    healthy = unit.health is not None and unit.health >= plan['min_health']
    if not state.engaged and (not healthy or target is None or ore is None):
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        return
    if state.engaged and (target is None or unit.health is None or unit.health < plan['retreat_health'] or
                          clock() - state.started >= plan['max_seconds']):
        state.returning = True
    if state.returning:
        if ore is not None:
            yield Act([Step.click(ore), Step.move(ctx.park)], commands=1)
            state.engaged = state.returning = False
            ctx.world.note('miner_support', status='harvest order sent', target=list(ore))
        else:
            yield Act([Step.key('s'), Step.move(ctx.park)], commands=1)
            ctx.world.note('miner_support', status='stopped; no fresh ore target')
    else:
        if not state.engaged:
            state.started = clock()
        state.engaged = True  # A cancelled input transaction must still trigger return handling.
        yield Act([Step.click(target, modifiers=('ctrl', 'shift')), Step.move(ctx.park)], commands=1)
        ctx.world.note('miner_support', status='defending', target=list(target))
    yield Act([Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=0)
    state.checked = clock()
