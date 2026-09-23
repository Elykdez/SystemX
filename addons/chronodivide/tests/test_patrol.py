from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2

from addons.chronodivide import patrol, playbook
from addons.chronodivide.executor import Trust
from addons.chronodivide.layout import Layout
from addons.chronodivide.perception.unitboxes import our_units
from addons.chronodivide.skills import Context
from addons.chronodivide.tasks import TaskRunner
from addons.chronodivide.world import World


def observed_army():
    canvas = cv2.imread(str(Path(__file__).with_name('fixtures') / 'army-battle.png'))
    return our_units(canvas)


def ready_world():
    world = World()
    world.set('credits', 4000)
    world.set('power', {'level': 'ok'})
    world.structures.update(construction_yard=1, ore_refinery=2, war_factory=1, barracks=1)
    world.set('radar', SimpleNamespace(online=True))
    return world


def test_squad_uses_existing_healthy_units_and_leaves_four_vehicles():
    units = observed_army()
    squad = patrol.choose_squad(units)
    assert [u.kind for u in squad].count('vehicle') == 3
    assert [u.kind for u in squad].count('infantry') == 1
    assert all(u.health >= 0.75 for u in squad)
    vehicles = [u for u in units if u.kind == 'vehicle' and u.health >= 0.75]
    infantry = next(u for u in units if u.kind == 'infantry' and u.health >= 0.75)
    assert not patrol.choose_squad(vehicles[:6] + [infantry])
    assert patrol.choose_squad(vehicles[:7] + [infantry])
    assert not patrol.choose_squad(vehicles[:7])


def test_names_inferred_from_build_counts_do_not_claim_requested_composition():
    units = [replace(u, name='rhino_tank', sure=True) for u in observed_army()]
    assert len(patrol.choose_squad(units)) == 4
    state = patrol.PatrolState()
    assert state.composition == '3 vehicles + 1 infantry; types unverified'


def test_forming_waits_for_economy_and_safety_and_never_replaces_automatically():
    world, state = ready_world(), patrol.PatrolState()
    assert patrol.can_form(world, state, 7, 31)
    assert not patrol.can_form(world, state, 6, 31)
    assert not patrol.can_form(world, state, 7, 5)
    world.set('credits', 1999)
    assert not patrol.can_form(world, state, 7, 31)
    world.set('credits', 4000)
    world.set('power', {'level': 'low'})
    assert not patrol.can_form(world, state, 7, 31)
    world.set('power', {'level': 'ok'})
    state.attempted = True
    assert not patrol.can_form(world, state, 20, 100)


def test_route_is_short_unique_and_in_the_home_view():
    ctx = Context(Layout.for_canvas(1024, 768), ready_world())
    route = patrol.guard_route(ctx, [(-500, 900), (-500, 900), (3000, 3000)])
    assert 2 <= len(route) <= 5 and len(route) == len(set(route))
    assert all(ctx.layout.tactical.contains(p) for p in route)


def test_contact_requests_existing_reinforcements_or_retreat():
    state = patrol.PatrolState(active=True)
    assert patrol.contact_action(state, True, 0.9) == 'reinforce'
    assert patrol.contact_action(state, True, 0.3) == 'retreat'
    assert patrol.contact_action(state, False, 0.9) is None
    state.active = False
    assert patrol.contact_action(state, True, 0.9) is None


def test_farther_scouting_requires_surplus_safe_home_and_healthy_squad():
    world, state = ready_world(), patrol.PatrolState(active=True)
    assert patrol.can_scout(world, state, 4, 60, 0.9)
    assert not patrol.can_scout(world, state, 3, 60, 0.9)
    assert not patrol.can_scout(world, state, 4, 15, 0.9)
    assert not patrol.can_scout(world, state, 4, 60, 0.6)
    world.set('credits', 3999)
    assert not patrol.can_scout(world, state, 4, 60, 0.9)


def run_formation(monkeypatch, selected_matches):
    time, batches = [10.0], []
    world = World(clock=lambda: time[0])
    ctx = Context(Layout.for_canvas(1024, 768), world)
    units = observed_army()
    squad = patrol.choose_squad(units)
    observations = iter([units, squad if selected_matches else units])
    monkeypatch.setattr(patrol, '_selected', lambda ctx: next(observations))
    executor = SimpleNamespace(act=lambda name, lane, steps, commands: batches.append(steps))
    runner = TaskRunner(executor, Trust(), clock=lambda: time[0])
    state = patrol.PatrolState()
    route = patrol.guard_route(ctx)
    runner.start('army:patrol', 'army', patrol.form_patrol(ctx, state, route))
    for _ in range(20):
        runner.tick(world.snapshot(), None)
        time[0] += 0.3
        if not runner.tasks:
            break
    assert not runner.tasks
    return state, batches


def test_verified_squad_queues_finite_route_and_releases_the_task_lane(monkeypatch):
    state, batches = run_formation(monkeypatch, True)
    steps = [s for batch in batches for s in batch]
    assert state.active and state.attempted and len(state.members) == 4
    assert any(s.kind == 'chord' and s.keys == ('ctrl', '3') for s in steps)
    route = next(s for s in steps if s.kind == 'route')
    assert route.keys == ('ctrl', 'shift') and len(route.points) == len(set(route.points))
    assert all(s.kind not in ('nudge',) for s in steps)


def test_failed_small_selection_cannot_assign_or_command_the_whole_army(monkeypatch):
    state, batches = run_formation(monkeypatch, False)
    steps = [s for batch in batches for s in batch]
    assert state.attempted and not state.active
    assert not any(s.kind in ('route', 'chord') for s in steps)


def scheduler(monkeypatch):
    from addons.chronodivide.army import Army
    from addons.chronodivide import army as army_module
    time, batches = [100.0], []
    world = ready_world()
    world.clock = lambda: time[0]
    world.set('mode', 'hud')
    world.set('radar', None)
    world.units_built['rhino_tank'] = 8
    ctx = Context(Layout.for_canvas(1024, 768), world)
    ctx.layout.points['waypoint'] = (362, 754)
    ctx.last_canvas = cv2.imread(str(Path(__file__).with_name('fixtures') / 'base-reactors.png'))
    executor = SimpleNamespace(act=lambda name, lane, steps, commands: batches.extend(steps))
    runner = TaskRunner(executor, Trust(), clock=lambda: time[0])
    army = Army(ctx, runner, clock=lambda: time[0])
    army.census_enabled = False
    army.record_census(observed_army())
    selected = [list(patrol.choose_squad(observed_army()))]
    monkeypatch.setattr(army_module, 'our_units', lambda *args, **kwargs: selected[0])
    return world, ctx, runner, army, time, batches, selected


def drain(runner, world, time):
    for _ in range(30):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.tasks:
            return
    raise AssertionError('a finite route or observation held the army lane')


def test_scheduler_forms_from_fresh_healthy_census_only_after_economy_recovers(monkeypatch):
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    world.set('credits', 1000)
    assert army.step(world.snapshot()) is None and not runner.tasks
    world.set('credits', 4000)
    time[0] += 2
    assert army.step(world.snapshot())[0] == 'army:patrol'
    assert runner.tasks['army'].name == 'army:patrol'
    assert world.units_built == {'rhino_tank': 8}  # no replacement purchases


def test_patrol_checks_request_fresh_defense_on_damage_without_firing(monkeypatch):
    from addons.chronodivide.army import monitor_patrol
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    army.patrol = patrol.PatrolState(attempted=True, active=True, members=tuple(selected[0]))
    selected[0] = [replace(u, health=0.6) for u in selected[0]]
    runner.start('army:patrol-check', 'army', monitor_patrol(ctx, army))
    drain(runner, world, time)
    assert army._recent_threat() and army.seen is None
    assert not any(s.kind == 'route' or (s.kind == 'click' and s.keys) for s in steps)
    assert army.step(world.snapshot()) == ('army:defend', None)


def test_patrol_refresh_waits_forty_seconds_and_uses_stop_then_finite_route(monkeypatch):
    from addons.chronodivide.army import monitor_patrol
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    army.patrol = patrol.PatrolState(attempted=True, active=True, members=tuple(selected[0]), last_order=70)
    runner.start('army:patrol-check', 'army', monitor_patrol(ctx, army))
    drain(runner, world, time)
    assert not any(s.kind == 'route' for s in steps)
    time[0] = 112
    runner.start('army:patrol-check', 'army', monitor_patrol(ctx, army))
    drain(runner, world, time)
    index = next(i for i, s in enumerate(steps) if s.kind == 'route')
    assert steps[index - 1].keys == ('s',)
    assert army.patrol.active and not runner.tasks


def test_partial_patrol_view_does_not_repeatedly_renew_defense(monkeypatch):
    from addons.chronodivide.army import monitor_patrol
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    members = tuple(selected[0])
    army.patrol = patrol.PatrolState(attempted=True, active=True, members=members)
    selected[0] = [members[-1]]
    for _ in range(3):
        runner.start('army:patrol-check', 'army', monitor_patrol(ctx, army))
        drain(runner, world, time)
        assert not army._recent_threat()
        time[0] += 8
    assert not army.patrol_verified and army.patrol.members == members
    assert not any(s.kind == 'route' for s in steps)


def test_missing_or_unhealthy_patrol_never_refreshes_route(monkeypatch):
    from addons.chronodivide.army import monitor_patrol
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    army.patrol = patrol.PatrolState(attempted=True, active=True, members=tuple(selected[0]))
    selected[0] = []
    runner.start('army:patrol-check', 'army', monitor_patrol(ctx, army))
    drain(runner, world, time)
    assert not army.patrol_verified and not any(s.kind == 'route' for s in steps)


def test_exploration_requires_observed_departure_and_return_not_queued_clicks(monkeypatch):
    from addons.chronodivide.army import monitor_exploration
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    infantry = next(u for u in observed_army() if u.kind == 'infantry' and u.health >= 0.75)
    army.exploring, army.explore_count = True, 1
    army.explore_home = infantry.center
    army.explore_origin = (infantry.center,)
    selected[0] = [infantry]
    def check():
        runner.start('army:explore-check', 'army', monitor_exploration(ctx, army))
        drain(runner, world, time)
    check()
    assert army.exploring and not army.explored
    selected[0] = [replace(infantry, box=(infantry.box[0] + 220, *infantry.box[1:]))]
    check()
    assert army.explore_departed and not army.explored
    selected[0] = [infantry]
    check()
    assert army.explored and not army.exploring


def test_exploration_deadline_is_failed_not_arrived(monkeypatch):
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    army.patrol.attempted = True
    army.exploring, army.explore_deadline = True, time[0] - 1
    army.step(world.snapshot())
    assert not army.explored and not army.exploring and army.explore_failed


def test_uncalibrated_waypoint_layout_skips_patrol(monkeypatch):
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    ctx.layout.points.pop('waypoint')
    assert army.step(world.snapshot()) is None and not runner.tasks


def test_mixed_patrol_scouts_only_near_home_with_surplus_and_avoids_known_enemy(monkeypatch):
    import numpy as np
    from addons.chronodivide.tests.test_strategy import online_radar
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, enemy=(), explored=np.zeros_like(radar.explored),
                              view=(home[0] - 25, home[1] - 25, home[0] + 25, home[1] + 25)))
    ctx.radar_home = home
    army.patrol.active = army.patrol_verified = True
    army.patrol_checked, army.patrol_health = time[0], 0.95
    point = army._scout_point(time[0])
    assert point is not None and abs(point[0] - home[0]) + abs(point[1] - home[1]) == 18
    army.intel = SimpleNamespace(enemy_base=point)
    other = army._scout_point(time[0])
    assert other is not None and other != point
    world.set('credits', 3999)
    assert army._scout_point(time[0]) is None
    world.set('credits', 4000)
    army.patrol_checked -= 13
    assert army._scout_point(time[0]) is None


def test_patrol_waits_for_healthy_infantry_without_consuming_formation_attempt(monkeypatch):
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    units = observed_army()
    vehicles = [u for u in units if u.kind == 'vehicle']
    infantry = next(u for u in units if u.kind == 'infantry' and u.health >= 0.75)
    for census in (vehicles, vehicles + [replace(infantry, health=0.4)]):
        army.record_census(census)
        time[0] += 2
        assert army.step(world.snapshot()) is None
        assert not army.patrol.attempted and not runner.tasks
    army.record_census(vehicles + [infantry])
    time[0] += 2
    assert army.step(world.snapshot())[0] == 'army:patrol'


def test_patrol_scout_uses_edge_of_actual_narrow_home_view(monkeypatch):
    import numpy as np
    from addons.chronodivide.tests.test_strategy import online_radar
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    radar = online_radar()
    world.set('radar', replace(radar, enemy=(), explored=np.zeros_like(radar.explored)))
    ctx.radar_home = radar.own[0][1]
    army.patrol.active = army.patrol_verified = True
    army.patrol_checked, army.patrol_health = time[0], 0.95
    point = army._scout_point(time[0])
    assert point is not None and radar.in_view(point, margin=-2)
    assert abs(point[0] - ctx.radar_home[0]) < 18 and abs(point[1] - ctx.radar_home[1]) < 18


def test_seven_node_exploration_allows_recorded_late_return_with_bounded_deadline(monkeypatch):
    from addons.chronodivide.army import explore_route, monitor_exploration
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    selected[0] = [u for u in observed_army() if u.kind == 'infantry' and u.health >= 0.75][:2]
    plan = army.book['intel']['explore']
    runner.start('army:explore', 'army', explore_route(ctx, army, '2', plan['route'], plan['leg_seconds']))
    drain(runner, world, time)
    route = next(s for s in steps if s.kind == 'route')
    assert len(route.points) == 7
    ordered_at = time[0]
    assert 139 <= army.explore_deadline - ordered_at <= 140
    time[0] = ordered_at + 90
    army.step(world.snapshot(), act=False)
    assert army.exploring and not army.explore_failed
    army.explore_departed = True
    x, y = army.explore_home
    selected[0] = [replace(u, box=(u.box[0] + x - u.center[0], u.box[1] + y - u.center[1],
                                  u.box[2] + x - u.center[0], u.box[3] + y - u.center[1]))
                   for u in selected[0]]
    time[0] = ordered_at + 110
    runner.start('army:explore-check', 'army', monitor_exploration(ctx, army))
    drain(runner, world, time)
    assert army.explored and not army.exploring and not army.explore_failed


def test_exploration_deadline_never_exceeds_three_minutes(monkeypatch):
    from addons.chronodivide.army import explore_route
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    selected[0] = [u for u in observed_army() if u.kind == 'infantry' and u.health >= 0.75][:2]
    runner.start('army:explore', 'army', explore_route(ctx, army, '2', [(200, 0), (0, 200)], 100))
    drain(runner, world, time)
    assert 179 <= army.explore_deadline - time[0] <= 180


def test_profile_controls_patrol_composition_reserve_economy_and_retreat():
    book = playbook.load()
    plan = book['patrol']
    plan.update(vehicles=2, infantry=2, reserve_vehicles=1, min_credits=4500,
                quiet_seconds=45, min_health=0.9, retreat_health=0.7)
    units = observed_army()
    vehicles = [replace(u, health=0.95) for u in units if u.kind == 'vehicle'][:3]
    infantry = [replace(u, health=0.95) for u in units if u.kind == 'infantry'][:2]
    squad = patrol.choose_squad(vehicles + infantry, book=book)
    assert [u.kind for u in squad] == ['vehicle', 'vehicle', 'infantry', 'infantry']
    assert not patrol.choose_squad([replace(u, health=0.8) for u in vehicles] + infantry, book=book)
    world, state = ready_world(), patrol.PatrolState()
    assert not patrol.can_form(world, state, 3, 50, book)
    world.set('credits', 4500)
    assert not patrol.can_form(world, state, 3, 44, book)
    assert patrol.can_form(world, state, 3, 45, book)
    assert not patrol.can_form(world, state, 2, 45, book)
    state.active = True
    assert patrol.contact_action(state, False, 0.65, book) == 'retreat'
    plan['scout'].update(min_credits=4500, quiet_seconds=45, min_health=0.9, min_refineries=3)
    assert not patrol.can_scout(world, state, 1, 45, 0.95, book)
    world.structures['ore_refinery'] = 3
    assert patrol.can_scout(world, state, 1, 45, 0.95, book)


def test_scheduler_uses_custom_patrol_health_counts_and_check_interval(monkeypatch):
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    army.book = playbook.load()
    army.book['patrol'].update(vehicles=2, infantry=2, reserve_vehicles=1, min_health=0.9,
                               check_every=20)
    units = observed_army()
    vehicles = [replace(u, health=0.95) for u in units if u.kind == 'vehicle'][:3]
    infantry = [replace(u, health=0.95) for u in units if u.kind == 'infantry'][:2]
    army.record_census(vehicles + infantry[:1])
    assert army.step(world.snapshot(), act=False) is None
    army.record_census(vehicles + infantry)
    time[0] += 2
    assert army.step(world.snapshot(), act=False)[0] == 'army:patrol'
    army.patrol.attempted = army.patrol.active = True
    army.patrol_checked = time[0]
    time[0] += 10
    assert army.step(world.snapshot(), act=False) is None
    time[0] += 10
    assert army.step(world.snapshot(), act=False)[0] == 'army:patrol-check'


def test_exploration_uses_profile_deadline_and_return_offset(monkeypatch):
    from addons.chronodivide.army import explore_route
    world, ctx, runner, army, time, steps, selected = scheduler(monkeypatch)
    army.book = playbook.load()
    plan = army.book['intel']['explore']
    plan.update(return_offset=[20, 80], deadline=dict(min_seconds=15, max_seconds=30, leg_multiplier=2))
    selected[0] = [u for u in observed_army() if u.kind == 'infantry' and u.health >= 0.75][:2]
    runner.start('army:explore', 'army', explore_route(ctx, army, '2', [(200, 0), (0, 200)], 100))
    drain(runner, world, time)
    assert 29 <= army.explore_deadline - time[0] <= 30
    assert army.explore_home == (ctx.home[0] + 20, ctx.home[1] + 80)
