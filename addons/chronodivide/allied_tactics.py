"""Prism siege and a nearby escort screen, using verified native type selections."""

import math

from .broker import Decision
from .executor import Step
from .perception.enemy_buildings import tesla_coil_targets
from .perception.unit_labels import read_label
from .perception.unitboxes import our_units
from .perception.units import enemy_blobs
from .tasks import Act, Look, Poll


def ground(unit):
    return unit.bar[0] + unit.bar[2] // 2, unit.bar[1] + 25


def separation(a, b):
    return math.hypot(a[0] - b[0], 2 * (a[1] - b[1]))


def offset(point, toward, reach, tactical):
    dx, dy = toward[0] - point[0], toward[1] - point[1]
    length = math.hypot(dx, 2 * dy)
    if length < 1:
        return None
    result = round(point[0] + reach * dx / length), round(point[1] + reach * dy / length)
    margin = 35
    return result if (tactical.x + margin <= result[0] <= tactical.x + tactical.w - margin and
                      tactical.y + margin <= result[1] <= tactical.y + tactical.h - margin) else None


def travel_step(campaign, center, toward, tactical, step, marker, count):
    key = (marker, campaign.target, campaign.waves, tuple(sorted(campaign.prism_roles.items())), count)
    previous = campaign.prism_travel
    stalls = (previous[2] + 1 if previous is not None and previous[0] == key
              and separation(previous[1], center) <= 12 else 0)
    # Ask the native pathfinder for an endpoint beyond local congestion only
    # after two acknowledged short moves have produced no visible progress.
    for scale in range(min(3, max(1, stalls)), 0, -1):
        point = offset(center, toward, step * scale, tactical)
        if point is not None:
            return point, (key, center, stalls)
    return None, None


def travelling(campaign, center, marker):
    previous = campaign.prism_move
    if previous is None:
        return False
    if previous[3] != (campaign.target, campaign.waves):
        campaign.prism_move = None
        return False
    if marker != previous[0] or separation(center, previous[1]) > 12:
        campaign.prism_move = (marker, center, True, previous[3])
        return True
    if previous[2]:
        campaign.prism_move = None
    return False


def escort_screen(center, toward, index, plan, tactical):
    front = offset(center, toward, plan['screen_pixels'], tactical)
    if front is None:
        return None
    dx, dy = toward[0] - center[0], toward[1] - center[1]
    side = -1 if index % 2 == 0 else 1
    flank = (front[0] - side * 2 * dy, front[1] + side * dx / 2)
    # Separate type groups leave the Prisms' next movement corridor open.
    return offset(front, flank, 2 * plan['screen_pixels'] * (1 + index // 2), tactical) or front


def staging_point(canvas, home, tactical, plan):
    from .army import building_point, BUILDING_PIXELS, MESSAGES
    region = (tactical.x, tactical.y, tactical.w, tactical.h)
    blobs = [b for b in enemy_blobs(canvas, region, min_area=25, close=3, compressed=True)
             if not MESSAGES.contains(b[1])]
    buildings = [building_point(canvas, b, compressed=True) for b in blobs if b[0] >= BUILDING_PIXELS]
    buildings = [p for p in buildings if p is not None]
    if not buildings:
        return None
    target = min(buildings, key=lambda p: separation(p, home))
    units = [b[1] for b in blobs if b[0] < BUILDING_PIXELS]
    for extra in (0, plan['step_pixels'], 2 * plan['step_pixels']):
        point = offset(target, home, plan['range_pixels'] + extra, tactical)
        if (point is not None and all(separation(point, b) >= plan['range_pixels'] - 2 for b in buildings)
                and all(separation(point, u) > plan['close_pixels'] for u in units)):
            return point
    return None


def approach_camera(radar, target, home, tactical, plan):
    if radar is None or not radar.online or radar.view is None or home is None:
        return target
    x0, y0, x1, y1 = radar.view
    dx, dy = home[0] - target[0], home[1] - target[1]
    length = math.hypot(dx * tactical.w / max(1, x1 - x0 + 1),
                        2 * dy * tactical.h / max(1, y1 - y0 + 1))
    if length < 1:
        return target
    fraction = plan['range_pixels'] / length
    return radar.inside((round(target[0] + dx * fraction), round(target[1] + dy * fraction)))


def selected(ctx):
    return [u for u in our_units(ctx.last_canvas, compressed=ctx.compressed_frames) if u.kind == 'vehicle']


def probe_units(ctx):
    tactical = ctx.layout.tactical
    # Hovering near the edge scrolls the camera and invalidates every later probe.
    return [u for u in selected(ctx)
            if tactical.x + 40 <= ground(u)[0] <= tactical.x + tactical.w - 40
            and tactical.y + 40 <= ground(u)[1] <= tactical.y + tactical.h - 40]


def select_role(ctx, team):
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key(team), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.15)
    return (yield Look(lambda s, v: selected(ctx)))


def center_role(ctx, team):
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key(team), Step.wait(80),
               Step.key(team), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.2)
    return (yield Look(lambda s, v: selected(ctx)))


def recover_escort(ctx, escort, prism_team):
    prisms = yield from center_role(ctx, prism_team)
    if prisms:
        yield Act([Step.key('s')])
    remote = yield from center_role(ctx, escort)
    if not remote:
        yield Poll(lambda s, v: False, 0.3)
        remote = yield Look(lambda s, v: selected(ctx))
    prisms = yield from center_role(ctx, prism_team)
    tactical = ctx.layout.tactical
    anchors = [ground(u) for u in prisms
               if tactical.x + 35 <= ground(u)[0] <= tactical.x + tactical.w - 35
               and tactical.y + 35 <= ground(u)[1] <= tactical.y + tactical.h - 35]
    if remote and anchors:
        # The escort was just confirmed in its own camera. Its bars are expected
        # to be offscreen after restoring the Prisms' current destination view.
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key(escort), Step.wait(150),
                   Step.click(anchors[0], modifiers=('alt',)), Step.move(ctx.park)])
        ctx.world.note('prism_escort_regroup', team=escort, visible=len(remote), point=anchors[0])
    return bool(remote)


def form_roles(ctx, campaign, plan):
    campaign.prism_move = None
    if ctx.camera == 'home':
        # A home-edge formation can hide escorts even though group 1 selects them.
        steps = [Step.click(ctx.clear_point, 'right'), Step.key('1'), Step.wait(80),
                 Step.key('1'), Step.wait(180), Step.click(ctx.clear_point, 'right'),
                 Step.key('p'), Step.wait(150), Step.key('s'), Step.key('1'), Step.wait(150)]
    else:
        # The battle session already found the front. A map-wide group centroid
        # can be far behind it while reinforcements are still traveling.
        steps = [Step.click(ctx.clear_point, 'right'), Step.key('p'), Step.wait(150), Step.key('s')]
    yield Act(steps + [Step.move(ctx.park)], commands=0)
    ctx.camera = 'away'
    yield Poll(lambda s, v: False, 0.2)
    units = yield Look(lambda s, v: probe_units(ctx))
    offset = campaign.prism_probe_offset % len(units) if units else 0
    probes = (units[offset:] + units[:offset])[:plan['label_probes']]
    # Retry beyond the same leading tanks so new units at the factory are inspected too.
    campaign.prism_probe_offset = (offset + len(probes)) % len(units) if units else 0
    for unit in probes:
        point = ground(unit)
        yield Act([Step.move(ctx.clear_point), Step.move(point), Step.wait(800)], commands=0)
        name = yield Look(lambda s, v: read_label(ctx.last_canvas, point))
        if name not in plan['teams'] or name in campaign.prism_roles:
            continue
        # Clear first: an empty-ground click must never move the previous selection.
        yield Act([Step.click(ctx.clear_point, 'right'), Step.click(point), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.15)
        picked = yield Look(lambda s, v: selected(ctx))
        if len(picked) != 1:
            continue
        yield Act([Step.move(ctx.clear_point), Step.move(ground(picked[0])), Step.wait(800)], commands=0)
        confirmed = yield Look(lambda s, v: read_label(ctx.last_canvas, ground(picked[0])))
        if confirmed != name:
            continue
        yield Act([Step.key('t'), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.15)
        typed = yield Look(lambda s, v: probe_units(ctx))
        if not typed:
            continue
        # T can extend across MAP when the visible type is already selected. Rebuild
        # from the visible bars so unseen home units never receive a siege order.
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        local = True
        for index, member in enumerate(typed):
            yield Act([Step.click(ground(member), modifiers=('shift',) if index else ()), Step.move(ctx.park)], commands=0)
            yield Poll(lambda s, v: False, 0.15)
            picked = yield Look(lambda s, v: selected(ctx))
            if len(picked) != index + 1:
                local = False
                break
            clicked = min(picked, key=lambda u: separation(ground(u), ground(member)))
            yield Act([Step.move(ctx.clear_point), Step.move(ground(clicked)), Step.wait(800)], commands=0)
            confirmed = yield Look(lambda s, v: read_label(ctx.last_canvas, ground(clicked)))
            if confirmed != name:
                local = False
                break
        if not local:
            yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
            continue
        team = plan['teams'][name]
        # Another lane may have cleared selection during the verification frames.
        steps = [Step.click(ctx.clear_point, 'right')]
        steps.extend(Step.click(ground(member), modifiers=('shift',) if index else ())
                     for index, member in enumerate(picked))
        yield Act(steps + [Step.chord('ctrl', team), Step.move(ctx.park)], commands=0)
        saved = yield from select_role(ctx, team)
        if len(saved) != len(typed):
            continue
        verified = True
        for member in saved:
            yield Act([Step.move(ctx.clear_point), Step.move(ground(member)), Step.wait(800)], commands=0)
            if (yield Look(lambda s, v: read_label(ctx.last_canvas, ground(member)))) != name:
                verified = False
                break
        if not verified:
            continue
        campaign.prism_roles[name] = team
        ctx.world.note('prism_role', item=name, team=team, visible=len(typed))


def picture(canvas, tactical, prisms, plan):
    from .army import army_center, building_point, BUILDING_PIXELS, MESSAGES, FIGHT_PIXELS, UNIT_SHARE
    if not prisms:
        return None
    center = army_center(prisms)
    region = (tactical.x, tactical.y, tactical.w, tactical.h)
    blobs = [b for b in enemy_blobs(canvas, region, min_area=25, close=3, compressed=True)
             if not MESSAGES.contains(b[1])]
    near = [b for b in blobs if separation(center, b[1]) <= plan['watch_pixels']]
    small = [b for b in near if b[0] < BUILDING_PIXELS]
    pixels = sum(b[0] for b in small)
    # Building trim splits into small house-color blobs too; reuse the army's
    # measured unit-share check before interpreting those fragments as attackers.
    units = ([b[1] for b in small] if pixels >= FIGHT_PIXELS and
             pixels >= UNIT_SHARE * sum(b[0] for b in near) else [])
    # Thin foundations can fall below the area cutoff. Keep these ambiguous
    # fragments in `units` too, so a close attacker still blocks bombardment.
    buildings = [building_point(canvas, b, compressed=True) for b in blobs
                 if b[0] >= BUILDING_PIXELS or (b[0] >= 60 and b[2][2] >= 40 and b[2][3] <= 25)]
    buildings.extend(tesla_coil_targets(canvas, region, compressed=True))
    buildings = [p for p in buildings if p is not None and
                 tactical.x + 35 <= p[0] <= tactical.x + tactical.w - 35 and
                 tactical.y + 35 <= p[1] <= tactical.y + tactical.h - 35]
    threat = min(units, key=lambda p: separation(center, p)) if units else None
    target = min(buildings, key=lambda p: separation(center, p)) if buildings else None
    approach = None
    if threat is not None and target is not None and separation(center, target) > plan['range_pixels']:
        # Leave a small margin for pixel rounding at the firing-distance boundary.
        reach = min(plan['step_pixels'], separation(center, target) - plan['range_pixels'] + 10)
        point = offset(center, target, reach, tactical)
        if point is not None and all(separation(point, unit) > plan['close_pixels'] for unit in units):
            approach = point
    return dict(center=center, target=target, threat=threat,
                close=threat is not None and separation(center, threat) <= plan['close_pixels'],
                hurt=any(u.health is None or u.health < plan['retreat_health'] for u in prisms),
                in_range=target is not None and separation(center, target) <= plan['range_pixels'],
                guarded_advance=approach)


def reveal_edge(canvas, tactical, units):
    """Look toward clipped enemy structures while keeping the visible formation in view."""
    from .army import BUILDING_PIXELS, MESSAGES
    if canvas is None or not units:
        return None
    # Compact enemy trim can occupy the message strip too; only wide text is excluded.
    blobs = [b for b in enemy_blobs(canvas, (tactical.x, tactical.y, tactical.w, tactical.h),
                                    min_area=BUILDING_PIXELS, compressed=True)
             if not MESSAGES.contains(b[1]) or b[2][2] <= 3 * b[2][3]]
    left, top = min(u.box[0] for u in units), min(u.box[1] for u in units)
    right, bottom = max(u.box[2] for u in units), max(u.box[3] for u in units)
    for _, _, (x, y, width, height) in blobs:
        if x <= tactical.x + 35 and right < tactical.x + tactical.w - 200:
            return 'Left'
        if x + width >= tactical.x + tactical.w - 35 and left > tactical.x + 200:
            return 'Right'
        if y <= tactical.y + 35 and bottom < tactical.y + tactical.h - 200:
            return 'Up'
        if y + height >= tactical.y + tactical.h - 35 and top > tactical.y + 200:
            return 'Down'
    return None


def choices(view, clear):
    if view is None:
        return ['hold']
    if view['close'] or view['hurt']:
        return ['reposition', 'hold']
    if not view.get('covered', False):
        return ['hold']
    options = ['hold']
    if view['in_range']:
        options.append('bombard')
    elif ((view['target'] is not None or view.get('route') is not None)
          and (view['threat'] is None or view.get('guarded_advance') is not None) and clear):
        options.append('advance')
    return options


def facts(view, clear):
    return dict(prisms='present' if view else 'missing',
                safety='danger' if view and (view['close'] or view['hurt']) else 'safe',
                target=('in_range' if view and view['in_range'] else 'distant' if view and view['target'] else
                        'route' if view and view.get('route') is not None else 'missing'),
                screen='clear' if clear else 'unconfirmed',
                cover='present' if view and view.get('covered') else 'missing')


def round_order(ctx, campaign, book, broker):
    plan = book['battle']['prism']

    def scene(units):
        view = picture(ctx.last_canvas, ctx.layout.tactical, units, plan)
        radar = ctx.world.radar.value
        if (view is not None and view['target'] is None and campaign.active and campaign.kind == 'enemy'
                and campaign.target is not None and radar is not None and radar.online and radar.view is not None
                and radar.enemy_near(campaign.target, 8)):
            # A remembered radar objective supplies direction, never a building to fire at.
            view = dict(view, route=radar.to_tactical(campaign.target, ctx.layout.tactical))
        return view

    if campaign.prism_role_wave != campaign.waves:
        campaign.prism_roles.clear()
        campaign.prism_probe_at = -1e9
        campaign.prism_probe_offset = 0
        campaign.prism_role_wave = campaign.waves
        campaign.prism_clear_since = None
        campaign.prism_move = None
    if (('prism_tank' not in campaign.prism_roles or len(campaign.prism_roles) < 2)
            and ctx.world.clock() - campaign.prism_probe_at >= 5):
        campaign.prism_probe_at = ctx.world.clock()
        yield from form_roles(ctx, campaign, plan)
    covered, escorts, recentered = False, [], False
    team = campaign.prism_roles.get('prism_tank')
    for name, escort in list(campaign.prism_roles.items()):
        if name != 'prism_tank':
            units = yield from select_role(ctx, escort)
            if units:
                covered = True
                escorts.extend(units)
            elif team:
                recentered = True
                if not (yield from recover_escort(ctx, escort, team)):
                    campaign.prism_roles.pop(name)
    if recentered:
        # Coordinates and coverage from the other camera cannot describe this view.
        covered, escorts = False, []
        for name, escort in campaign.prism_roles.items():
            if name != 'prism_tank':
                units = yield from select_role(ctx, escort)
                covered = covered or bool(units)
                escorts.extend(units)
    prisms = (yield from select_role(ctx, team)) if team else []
    view = yield Look(lambda s, v: scene(prisms))
    if view is not None and view['target'] is None:
        direction = reveal_edge(ctx.last_canvas, ctx.layout.tactical, prisms + escorts)
        if direction is not None:
            yield Act([Step.key(direction), Step.move(ctx.park)], commands=0)
            yield Poll(lambda s, v: False, 0.2)
            prisms = yield from select_role(ctx, team)
            view = yield Look(lambda s, v: scene(prisms))
            campaign.prism_clear_since = None
            ctx.world.note('prism_camera', direction=direction, reason='clipped enemy structures')
    if view is not None:
        view['covered'] = covered
    now = ctx.world.clock()
    radar = ctx.world.radar.value
    marker = radar.view if radar is not None else None
    if marker != campaign.prism_view:
        campaign.prism_clear_since = None
        campaign.prism_view = marker
    if view is None or (view['threat'] is not None and view.get('guarded_advance') is None):
        campaign.prism_clear_since = None
    elif campaign.prism_clear_since is None:
        campaign.prism_clear_since = now
    clear = campaign.prism_clear_since is not None and now - campaign.prism_clear_since >= plan['clear_seconds']
    options = choices(view, clear)
    classes = facts(view, clear)
    default = book.rule('prism_battle', options, classes)
    answer = {'choice': default, 'source': 'rule'}
    if broker is not None and len(options) > 1:
        words = {key: book.say('prism_battle', key, value) for key, value in classes.items()}
        def decided(decision, answers, provenance, meta):
            choice = (answers or {}).get('prism_battle', {}).get('choice')
            if choice in options:
                answer.update(choice=choice, source=provenance)
        broker.submit(Decision('tactical:prism_battle', 'tactical', words, book.question('prism_battle', options),
                               broker.clock() + 0.6, decided, {'prism_battle': {'choice': default}},
                               generation=broker.generation()))
        yield Poll(lambda s, v: answer['source'] != 'rule', 0.7)
    # Recheck after the model's wait; it may no longer be safe to advance or fire.
    if team:
        view = yield Look(lambda s, v: scene(selected(ctx)))
        if view is not None:
            view['covered'] = covered
    else:
        view = None
    option = answer['choice'] if answer['choice'] in choices(view, clear) else 'hold'
    safe_travel = (view is not None and not view['in_range']
                   and (view['target'] is not None or view.get('route') is not None)
                   and (view['threat'] is None or view.get('guarded_advance') is not None)
                   and not view['close'] and not view['hurt'] and view['covered'])
    continuing = False
    if safe_travel and (option == 'advance' or (option == 'hold' and not clear and answer['source'] == 'rule')):
        continuing = travelling(campaign, view['center'], marker)
    else:
        campaign.prism_move = None
    if not continuing and (option != 'advance' or view is None or view['target'] is not None):
        campaign.prism_travel = None
    orders = []
    if view is not None:
        center, target, threat = view['center'], view['target'], view['threat']
        if continuing:
            # Replacing the endpoint during a detour can send the group back
            # across the same obstacle. Let the acknowledged move finish.
            pass
        elif option == 'bombard':
            yield Act([Step.click(target, modifiers=('ctrl',)), Step.move(ctx.park)])
            orders.append(('prism_bombard', target))
        elif option in ('advance', 'reposition'):
            toward = (target or view.get('route')) if option == 'advance' else (
                2 * center[0] - (threat or target or center)[0],
                2 * center[1] - (threat or target or center)[1])
            travel = None
            if option == 'advance' and target is None:
                point, travel = travel_step(campaign, center, toward, ctx.layout.tactical,
                                           plan['step_pixels'], marker, len(selected(ctx)))
            elif option == 'advance' and view.get('guarded_advance') is not None:
                point = view['guarded_advance']
            else:
                point = offset(center, toward, plan['step_pixels'], ctx.layout.tactical)
            if point is not None:
                # A destination occupied by a friendly unit must remain a move order.
                yield Act([Step.click(point, modifiers=('alt',)), Step.move(ctx.park)])
                campaign.prism_travel = travel
                campaign.prism_move = ((marker, center, False, (campaign.target, campaign.waves))
                                       if option == 'advance' else None)
                orders.append(('prism_' + option, point))
            else:
                yield Act([Step.key('s')])
        else:
            yield Act([Step.key('s')])
        # Escorts screen near the Prisms. Only a close threat permits attack-move.
        escort_teams = [escort for name, escort in campaign.prism_roles.items() if name != 'prism_tank']
        for index, escort in enumerate(escort_teams):
            screen = escort_screen(center, threat or target or view.get('route') or center,
                                   index, plan, ctx.layout.tactical)
            units = yield from select_role(ctx, escort)
            if not units:
                continue
            fresh = yield Look(lambda s, v: picture(ctx.last_canvas, ctx.layout.tactical, prisms, plan))
            close = fresh is not None and fresh['close']
            if close:
                yield Act([Step.click(fresh['threat'], modifiers=('ctrl', 'shift')), Step.move(ctx.park)])
                orders.append(('escort_defend', fresh['threat']))
            elif screen is not None:
                from .army import army_center
                if separation(army_center(units), screen) > plan['screen_pixels'] / 3:
                    yield Act([Step.click(screen, modifiers=('alt',)), Step.move(ctx.park)])
                    orders.append(('escort_screen', screen))
                else:
                    yield Act([Step.key('s')])
    else:
        # One P after clearing selects military units on SCREEN. Only stop when
        # visible bars confirm the selection; unseen arrivals keep travelling.
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key('p'), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.15)
        units = yield Look(lambda s, v: selected(ctx))
        if units:
            yield Act([Step.key('s'), Step.move(ctx.park)])
    campaign.sieging = bool(view is not None and view['target'] is not None and not view['hurt'] and covered)
    ctx.world.note('prism_decision', choice=option, source=answer['source'], facts=classes,
                   roles=dict(campaign.prism_roles), orders=orders, continuing=continuing)
    return orders
