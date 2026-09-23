"""Small, existing-force guard squad. The army scheduler owns monitoring and refreshes.

Routes are finite: Chrono Divide rejects duplicate waypoint nodes. No helper buys
units, replaces a lost squad, or holds the task lane while waiting for its arrival.
"""

from dataclasses import dataclass, field

from . import playbook
from .executor import Step
from .perception.unitboxes import our_units
from .tasks import Act, Look, Poll

def _plan(book=None):
    return (book or playbook.default())['patrol']


def _composition(plan):
    return f"{plan['vehicles']} vehicles + {plan['infantry']} infantry; types unverified"


@dataclass
class PatrolState:
    attempted: bool = False
    active: bool = False
    members: tuple = ()
    route: tuple = ()
    last_order: float = 0.0
    composition: str = field(default_factory=lambda: _composition(_plan()))
    contact: tuple | None = None


def can_form(world, state, visible_vehicles, threat_age, book=None):
    """Spend nothing and preserve the configured reserve; never reform after losses."""
    plan = _plan(book)
    credits = world.spendable()
    power = world.power.value or {}
    return (not state.attempted and credits is not None and credits >= plan['min_credits'] and
            visible_vehicles >= plan['reserve_vehicles'] + plan['vehicles'] and
            threat_age >= plan['quiet_seconds'] and power.get('level') == 'ok' and
            all(world.structures.get(item, 0) for item in plan['requires']))


def choose_squad(units, reserve=None, book=None):
    """Choose only P-selected military units with health bars; names from build counts are guesses."""
    plan = _plan(book)
    reserve = plan['reserve_vehicles'] if reserve is None else reserve
    healthy = [u for u in units if u.side == 'ours' and u.bar is not None and
               u.health is not None and u.health >= plan['min_health']]
    vehicles = [u for u in healthy if u.kind == 'vehicle']
    infantry = [u for u in healthy if u.kind == 'infantry']
    if len(vehicles) < reserve + plan['vehicles'] or len(infantry) < plan['infantry']:
        return ()
    # Cargo pips suggest a transport, but do not prove the requested Flak Track type.
    cargo = next((u for u in vehicles if u.cargo), None) if plan['prefer_transport'] else None
    selected = ([cargo] if cargo else [])
    selected += sorted((u for u in vehicles if u is not cargo), key=lambda u: -u.health)[:plan['vehicles'] - len(selected)]
    return tuple(selected + sorted(infantry, key=lambda u: -u.health)[:plan['infantry']])


def guard_route(ctx, approaches=(), book=None):
    """A short route around the base plus caller-observed mining approaches, in the home view."""
    plan = _plan(book)['route']
    x, y = ctx.front_point(reach=plan['front_reach'])
    tactical = ctx.layout.tactical
    candidates = [(x + dx, y + dy) for dx, dy in plan['offsets']]
    candidates[1:1] = list(approaches)[:plan['max_approaches']]
    points = []
    for px, py in candidates:
        point = (round(min(max(px, tactical.x + 45), tactical.x + tactical.w - 45)),
                 round(min(max(py, tactical.y + 45), tactical.y + tactical.h - 45)))
        if point not in points:
            points.append(point)
    return tuple(points)


def can_scout(world, state, visible_reserve, threat_age, squad_health, book=None):
    """Farther scouting waits for a healthy economy, squad and home reserve."""
    plan = _plan(book)
    scout = plan['scout']
    credits = world.spendable()
    radar = world.radar.value
    return (state.active and credits is not None and credits >= scout['min_credits'] and
            visible_reserve >= plan['reserve_vehicles'] and threat_age >= scout['quiet_seconds'] and
            squad_health is not None and squad_health >= scout['min_health'] and
            world.structures.get(scout['refinery'], 0) >= scout['min_refineries'] and
            (world.power.value or {}).get('level') == 'ok' and radar is not None and radar.online)


def contact_action(state, enemy_near, squad_health, book=None):
    """A bounded signal for the scheduler; reinforcements are existing defenders, never purchases."""
    if not state.active:
        return None
    if squad_health is not None and squad_health < _plan(book)['retreat_health']:
        return 'retreat'
    return 'reinforce' if enemy_near else None


def _selected(ctx):
    return our_units(ctx.last_canvas, compressed=ctx.compressed_frames) if ctx.last_canvas is not None else []


def _same_selection(selected, expected):
    if len(selected) != len(expected):
        return False
    remaining = list(selected)
    for unit in expected:
        match = next((u for u in remaining if u.kind == unit.kind and
                      abs(u.center[0] - unit.center[0]) + abs(u.center[1] - unit.center[1]) <= 36), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def form_patrol(ctx, state, route, book=None):
    """One attempt: select a small squad, verify it from a fresh frame, then save team 3."""
    if state.attempted:
        return
    state.attempted = True
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    yield Act([Step.key('p'), Step.wait(150), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.12)
    plan = _plan(book)
    squad = choose_squad((yield Look(lambda s, v: _selected(ctx))), book=book)
    if not squad:
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        ctx.world.note('patrol', grouped=False, reason='insufficient verified squad and home reserve')
        return
    steps = [Step.click(ctx.clear_point, 'right')]
    for index, unit in enumerate(squad):
        steps.extend([Step.click(unit.center, modifiers=('shift',) if index else ()), Step.wait(80)])
    yield Act(steps + [Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.15)
    selected = yield Look(lambda s, v: _selected(ctx))
    if not _same_selection(selected, squad):
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        ctx.world.note('patrol', grouped=False, reason='small squad selection not verified')
        return
    yield Act([Step.chord('ctrl', plan['team']), Step.wait(80)], commands=0)
    state.active, state.members = True, tuple(selected)
    state.composition = _composition(plan)
    ctx.world.note('patrol', grouped=True, composition=state.composition, team=plan['team'])
    yield from order_patrol(ctx, state, route, at_home=True, book=book)


def order_patrol(ctx, state, route, at_home=False, book=None):
    """Queue one finite route and release the lane; refresh only after fresh monitoring."""
    if not state.active or not route:
        return
    if not at_home:
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.25)
        ctx.camera = 'home'
    point = ctx.layout.points.get('waypoint', (362, ctx.layout.canvas[1] - 14))
    team = _plan(book)['team']
    yield Act([Step.key(team), Step.wait(120), Step.key('s'), Step.route(route, point, attack_move=True),
               Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=len(route))
    state.route, state.last_order = tuple(route), ctx.world.clock()
    ctx.world.note('order', order='patrol', team=team, waypoints=len(route), finite=True)
