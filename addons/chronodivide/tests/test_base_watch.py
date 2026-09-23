from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.army import Army, HOME_RADIUS, base_watch, event_watch
from addons.chronodivide import playbook
from addons.chronodivide.executor import Trust
from addons.chronodivide.layout import Layout
from addons.chronodivide.perception.radar import distance
from addons.chronodivide.skills import Context
from addons.chronodivide.tasks import TaskRunner
from addons.chronodivide.world import Slot, World
from addons.chronodivide.tests.test_strategy import online_radar


def setup(peek='base-reactors.png', radar_contact=False):
    time, steps = [100.0], []
    world = World(clock=lambda: time[0])
    world.set('mode', 'hud')
    world.units_built['rhino_tank'] = 3
    radar = replace(online_radar(), enemy=())
    world.set('radar', radar)
    ctx = Context(Layout.for_canvas(1024, 768), world)
    ctx.radar_home = radar.own[0][1]
    fixtures = Path(__file__).with_name('fixtures')
    home_frame = cv2.imread(str(fixtures / 'base-reactors.png'))
    ctx.last_canvas = home_frame
    contact = (ctx.radar_home[0] + 6, ctx.radar_home[1] + 4)

    class Camera:
        def act(self, name, lane, batch, commands=1):
            steps.extend(batch)
            for step in batch:
                if (step.kind == 'click' and step.button == 'left'
                        and ctx.layout.regions['radar'].contains(step.point)):
                    x, y = step.point
                    world.set('radar', replace(radar, view=(x - 8, y - 14, x + 9, y + 14),
                                              enemy=((12, contact),) if radar_contact else ()))
                    ctx.last_canvas = cv2.imread(str(fixtures / peek))
                if step.kind == 'key' and step.keys == ('h',):
                    world.set('radar', replace(radar, enemy=((12, contact),) if radar_contact else ()))
                    ctx.last_canvas = home_frame

    runner = TaskRunner(Camera(), Trust(), clock=lambda: time[0])
    book = playbook.load()
    # This camera simulator does not render the selections needed to split infantry from vehicles.
    book['attack']['keep_home_infantry'] = False
    army = Army(ctx, runner, clock=lambda: time[0], book=book)
    army.census_enabled = False
    return world, ctx, runner, army, time, steps


def drain(world, runner, time):
    for _ in range(80):
        runner.tick(world.snapshot(), None)
        time[0] += 0.1
        if not runner.tasks:
            return
    raise AssertionError('camera sweep did not release its lane')


def test_periodic_sweep_clears_selection_and_only_moves_camera():
    world, ctx, runner, army, time, steps = setup()
    assert army.step(world.snapshot()) is None
    time[0] += 26
    name, point = army.step(world.snapshot())
    assert name == 'army:base-watch' and distance(point, ctx.radar_home) <= HOME_RADIUS
    assert world.radar.value.explored_near(point, radius=2) >= 0.6
    drain(world, runner, time)
    clicks = [s for s in steps if s.kind == 'click']
    assert clicks[0].button == 'right' and clicks[1].button == 'left' and clicks[1].point == point
    assert [s.keys for s in steps if s.kind == 'key'] == [('h',)]
    assert not any(s.kind in ('route', 'chord') or s.keys for s in clicks)
    assert ctx.camera == 'home' and not any(e['kind'] == 'attack' for e in world.events)
    assert army._watch_point() != point
    time[0] += 2
    assert army.step(world.snapshot()) is None


def test_defensive_posture_still_inspects_approaches_during_a_quiet_interval():
    from addons.chronodivide.intent import StandingIntent
    world, ctx, runner, army, time, steps = setup()
    army.intent = StandingIntent()
    army.intent.set('posture', 'defend', 'test')
    time[0] += 26
    assert army.step(world.snapshot())[0] == 'army:base-watch'
    drain(world, runner, time)
    assert ctx.camera == 'home'


def test_visible_contact_prioritizes_fresh_home_defense_without_storing_a_fire_point():
    world, ctx, runner, army, time, steps = setup(peek='enemy-near-base.png')
    runner.start('army:base-watch', 'army', base_watch(ctx, army, army._watch_point()))
    drain(world, runner, time)
    assert any(e.get('source') == 'base_watch' and e['pixels'] >= 50 for e in world.events)
    assert army.step(world.snapshot()) == ('army:defend', None)
    drain(world, runner, time)
    assert sum(s.kind == 'click' and s.keys == ('ctrl', 'shift') for s in steps) == 2


def test_radar_contact_responds_to_current_enemy_point_not_camera_destination():
    world, ctx, runner, army, time, steps = setup(radar_contact=True)
    destination = army._watch_point()
    runner.start('army:base-watch', 'army', base_watch(ctx, army, destination))
    drain(world, runner, time)
    contact = world.radar.value.enemy[0][1]
    assert contact != destination
    assert army.step(world.snapshot()) == ('army:defend-radar', contact)


@pytest.mark.parametrize('profile', ['startlux-ra2-allied.json', 'startlux-ra2-soviet.json'])
@pytest.mark.parametrize('enemy_visible', [False, True])
def test_sweep_follows_a_contact_outside_the_inspected_approach_before_defending(profile, enemy_visible):
    world, ctx, runner, army, time, steps = setup(radar_contact=True)
    army.book = playbook.load(Path(__file__).parents[1] / 'profiles' / profile)
    army.book['attack']['keep_home_infantry'] = False
    home = ctx.radar_home
    destination, contact = (home[0]-8, home[1]-10), (home[0]+6, home[1]+4)
    original = runner.executor.act

    def reveal_contact(name, lane, batch, commands=1):
        original(name, lane, batch, commands)
        if enemy_visible and any(s.kind == 'click' and s.button == 'left' and s.point == contact for s in batch):
            ctx.last_canvas = cv2.imread(str(Path(__file__).with_name('fixtures') / 'enemy-near-base.png'))

    runner.executor.act = reveal_contact
    runner.start('army:base-watch', 'army', base_watch(ctx, army, destination, rounds=1))
    drain(world, runner, time)
    camera_clicks = [s.point for s in steps if s.kind == 'click' and s.button == 'left'
                     and ctx.layout.regions['radar'].contains(s.point)]
    assert camera_clicks == [destination, contact]
    orders = [i for i, s in enumerate(steps) if s.kind == 'click' and s.keys == ('ctrl', 'shift')]
    assert bool(orders) == enemy_visible
    if orders:
        focus = next(i for i, s in enumerate(steps) if s.kind == 'click' and s.point == contact)
        assert focus < min(orders)
    assert ctx.camera == 'home'


@pytest.mark.parametrize('profile', ['startlux-ra2-allied.json', 'startlux-ra2-soviet.json'])
def test_offscreen_defense_keeps_camera_on_fresh_raiders_for_the_session(profile):
    world, ctx, runner, army, time, steps = setup(peek='enemy-near-base.png', radar_contact=True)
    army.book = playbook.load(Path(__file__).parents[1] / 'profiles' / profile)
    army.book['attack']['keep_home_infantry'] = False
    world.units_built[army.book['units']['tank']] = 3
    contact = (ctx.radar_home[0] + 6, ctx.radar_home[1] + 4)
    world.set('radar', replace(world.radar.value, enemy=((12, contact),),
                            view=(ctx.radar_home[0]-4, ctx.radar_home[1]-4,
                                  ctx.radar_home[0]+4, ctx.radar_home[1]+4)))
    army.watch_contact_at = time[0]
    assert army.step(world.snapshot()) == ('army:defend-radar', contact)
    for _ in range(300):
        runner.tick(world.snapshot(), None)
        time[0] += .1
        if not runner.tasks:
            break
    assert not runner.tasks and ctx.camera == 'home'
    orders = [i for i, s in enumerate(steps) if s.kind == 'click' and s.keys == ('ctrl', 'shift')]
    home = next(i for i, s in enumerate(steps) if s.kind == 'key' and s.keys == ('h',))
    assert len(orders) == army.book['defense']['session_rounds'] and max(orders) < home
    assert all(not ctx.layout.regions['radar'].contains(steps[i].point) for i in orders)


def test_offscreen_defense_does_not_fire_when_the_revealed_view_has_no_enemy():
    world, ctx, runner, army, time, steps = setup(radar_contact=True)
    contact = (ctx.radar_home[0]+6, ctx.radar_home[1]+4)
    runner.start('army:defend', 'army', base_watch(ctx, army, contact, rounds=8))
    drain(world, runner, time)
    assert ctx.camera == 'home'
    assert not any(s.kind == 'click' and s.keys == ('ctrl', 'shift') for s in steps)


@pytest.mark.parametrize('profile', ['startlux-ra2-allied.json', 'startlux-ra2-soviet.json'])
@pytest.mark.parametrize('home_vehicles_visible', [False, True])
def test_offscreen_defense_forms_vehicle_team_and_leaves_home_infantry(profile, home_vehicles_visible):
    from addons.chronodivide.perception.unitboxes import our_units
    from addons.chronodivide.tests.test_unitboxes import draw_bar
    world, ctx, runner, army, time, steps = setup(peek='enemy-near-base.png', radar_contact=True)
    army.book = playbook.load(Path(__file__).parents[1] / 'profiles' / profile)
    assert army.book['attack']['keep_home_infantry']
    original = runner.executor.act
    removed = []

    def select_defenders(name, lane, batch, commands=1):
        original(name, lane, batch, commands)
        if ctx.camera == 'home' and any(s.kind == 'key' and s.keys == ('p',) for s in batch):
            draw_bar(ctx.last_canvas, 200, 200, 18)
            if home_vehicles_visible:
                draw_bar(ctx.last_canvas, 400, 200, 36)
        for step in batch:
            if step.kind == 'click' and step.keys == ('shift',):
                infantry = [u for u in our_units(ctx.last_canvas) if u.kind == 'infantry']
                if any((u.bar[0] + u.bar[2] // 2, u.bar[1] + 25) == step.point for u in infantry):
                    ctx.last_canvas[195:240, 180:235] = 0
                    removed.append(step.point)

    runner.executor.act = select_defenders
    contact = (ctx.radar_home[0]+6, ctx.radar_home[1]+4)
    runner.start('army:defend', 'army', base_watch(ctx, army, contact, rounds=8))
    for _ in range(300):
        runner.tick(world.snapshot(), None)
        time[0] += .1
        if not runner.tasks:
            break
    assert not runner.tasks and ctx.camera == 'home' and len(removed) == 1
    assert sum(s.kind == 'chord' and s.keys == ('ctrl', '1') for s in steps) == 1
    assert sum(s.kind == 'key' and s.keys == ('1',) for s in steps) == 16
    orders = [i for i, s in enumerate(steps) if s.kind == 'click' and s.keys == ('ctrl', 'shift')]
    homes = [i for i, s in enumerate(steps) if s.kind == 'key' and s.keys == ('h',)]
    assert len(orders) == 8 and homes[0] < min(orders) < max(orders) < homes[-1]


@pytest.mark.parametrize('defenders_visible', [False, True])
def test_offscreen_raider_gets_fresh_attack_move_before_camera_returns_home(defenders_visible):
    from addons.chronodivide.tests.test_unitboxes import draw_bar
    world, ctx, runner, army, time, steps = setup(peek='enemy-near-base.png')
    original = runner.executor.act

    def select_defenders(name, lane, batch, commands=1):
        original(name, lane, batch, commands)
        if defenders_visible and any(step.kind == 'key' and step.keys == ('p',) for step in batch):
            draw_bar(ctx.last_canvas, 300, 400, 36, 0.9)

    runner.executor.act = select_defenders
    runner.start('army:base-watch', 'army', base_watch(ctx, army, army._watch_point()))
    drain(world, runner, time)
    orders = [i for i, step in enumerate(steps) if step.kind == 'click' and step.keys == ('ctrl', 'shift')]
    home = next(i for i, step in enumerate(steps) if step.kind == 'key' and step.keys == ('h',))
    assert len(orders) == 2 and max(orders) < home
    assert ctx.camera == 'home'
    assert not any(step.kind == 'click' and step.keys == ('ctrl',) for step in steps)


def test_sweep_waits_for_construction_and_never_peeks_into_shroud():
    world, ctx, runner, army, time, steps = setup()
    time[0] += 26
    world.update_sidebar('buildings', [Slot(0, 'war_factory', 'ready', 1.0)])
    assert army.step(world.snapshot()) is None and not runner.tasks
    radar = world.radar.value
    world.set('radar', replace(radar, explored=np.zeros_like(radar.explored)))
    assert army._watch_point() is None


def test_ready_building_during_peek_returns_home_and_releases_lane():
    world, ctx, runner, army, time, steps = setup()
    runner.start('army:base-watch', 'army', base_watch(ctx, army, army._watch_point()))
    runner.tick(world.snapshot(), None)
    world.update_sidebar('buildings', [Slot(0, 'war_factory', 'ready', 1.0)])
    drain(world, runner, time)
    assert ctx.camera == 'home' and time[0] - 100 < 1


def test_cancelled_peek_marks_camera_lost_and_next_order_restores_home():
    world, ctx, runner, army, time, steps = setup()
    runner.start('army:base-watch', 'army', base_watch(ctx, army, army._watch_point()))
    runner.tick(world.snapshot(), None)
    runner.cancel_all('paused during camera move')
    assert ctx.camera == 'lost'
    assert army.step(world.snapshot()) == ('army:camera', None)
    drain(world, runner, time)
    assert ctx.camera == 'home' and steps[-1].keys == ('h',)


def test_census_and_existing_army_tasks_take_priority_over_camera_sweeps():
    world, ctx, runner, army, time, steps = setup()
    time[0] += 26
    army.census_enabled = True
    assert army.step(world.snapshot())[0] == 'army:census'
    assert runner.tasks['army'].name == 'army:census'


@pytest.mark.parametrize('profile', ['startlux-ra2-allied.json', 'startlux-ra2-soviet.json'])
def test_offscreen_threat_is_inspected_before_census_even_without_defenders(profile):
    world, ctx, runner, army, time, steps = setup(radar_contact=True)
    army.book = playbook.load(Path(__file__).parents[1] / 'profiles' / profile)
    world.units_built.clear()
    army.census_enabled = True
    contact = (ctx.radar_home[0]+6, ctx.radar_home[1]+4)
    world.set('radar', replace(world.radar.value, enemy=((12, contact),),
                            view=(ctx.radar_home[0]-4, ctx.radar_home[1]-4,
                                  ctx.radar_home[0]+4, ctx.radar_home[1]+4)))
    assert army.step(world.snapshot()) == ('army:defend-radar', contact)
    assert runner.tasks['army'].name == 'army:defend'


def test_periodic_watch_inspects_offscreen_own_base_contact_and_records_fresh_survey():
    from addons.chronodivide.intel import Intel
    world, ctx, runner, army, time, steps = setup()
    army.intel = Intel(lambda: time[0], book=army.book)
    point = (ctx.radar_home[0]+6, ctx.radar_home[1]+4)
    world.set('radar', replace(world.radar.value, own=((20, point),),
                            view=(ctx.radar_home[0]-4, ctx.radar_home[1]-4,
                                  ctx.radar_home[0]+4, ctx.radar_home[1]+4)))
    assert army._watch_point() == point
    runner.start('army:base-watch', 'army', base_watch(ctx, army, point))
    drain(world, runner, time)
    assert army.intel.looks['base']['point'] == list(point)
    assert army.intel.looks['base']['at'] > 100
    assert not world.structures


@pytest.mark.parametrize('kind', ['attack', 'structure-lost'])
def test_incident_without_radar_is_inspected_before_census_once(kind):
    world, ctx, runner, army, time, steps = setup()
    world.set('radar', None)
    army.census_enabled = True
    time[0] += 26
    world.note(kind, item='war_factory')
    army.threat_at = time[0] if kind == 'attack' else -1e9
    assert army.step(world.snapshot()) == ('army:event-watch', None)
    drain(world, runner, time)
    assert ('space',) in [s.keys for s in steps]
    assert ctx.camera == 'home'
    time[0] += 9
    assert army.step(world.snapshot())[0] != 'army:event-watch'


@pytest.mark.parametrize('peek', ['base-reactors.png', 'enemy-near-base.png'])
def test_event_inspection_uses_fresh_pixels_and_does_not_invent_a_target(peek):
    from addons.chronodivide.intel import Intel
    world, ctx, runner, army, time, steps = setup()
    world.set('radar', None)
    world.units_built.clear()
    army.intel = Intel(lambda: time[0], book=army.book)
    original = runner.executor.act

    def incident_camera(name, lane, batch, commands=1):
        original(name, lane, batch, commands)
        if any(s.kind == 'key' and s.keys == ('space',) for s in batch):
            ctx.last_canvas = cv2.imread(str(Path(__file__).with_name('fixtures') / peek))

    runner.executor.act = incident_camera
    runner.start('army:event-watch', 'army', event_watch(ctx, army))
    drain(world, runner, time)
    assert ctx.camera == 'home'
    assert army.intel.looks['base']['point'] == []
    assert bool(army.intel.looks['base']['enemy_pixels']) == (peek == 'enemy-near-base.png')
    assert any(e['kind'] == 'attack' for e in world.events) == (peek == 'enemy-near-base.png')
    assert not any(s.kind == 'click' and s.keys for s in steps)
    assert not world.structures


def test_stale_incident_without_radar_does_not_move_camera():
    world, ctx, runner, army, time, steps = setup()
    world.set('radar', None)
    world.note('structure-lost', item='war_factory')
    time[0] += 26
    assert army.step(world.snapshot()) is None
    assert not steps


@pytest.mark.parametrize('home_vehicles_visible', [False, True])
def test_event_defense_without_radar_preserves_home_infantry_and_fights_before_return(home_vehicles_visible):
    from addons.chronodivide.perception.unitboxes import our_units
    from addons.chronodivide.tests.test_unitboxes import draw_bar
    world, ctx, runner, army, time, steps = setup()
    world.set('radar', None)
    army.book['attack']['keep_home_infantry'] = True
    original = runner.executor.act
    removed = []

    def incident_camera(name, lane, batch, commands=1):
        original(name, lane, batch, commands)
        for step in batch:
            if step.keys == ('space',):
                ctx.last_canvas = cv2.imread(str(Path(__file__).with_name('fixtures') / 'enemy-near-base.png'))
            if step.kind == 'key' and step.keys == ('p',):
                draw_bar(ctx.last_canvas, 200, 200, 18)
                if home_vehicles_visible:
                    draw_bar(ctx.last_canvas, 400, 200, 36)
            if step.kind == 'key' and step.keys == ('1',):
                draw_bar(ctx.last_canvas, 300, 400, 36)
            if step.kind == 'click' and step.keys == ('shift',):
                infantry = [u for u in our_units(ctx.last_canvas) if u.kind == 'infantry']
                if any((u.bar[0] + u.bar[2] // 2, u.bar[1] + 25) == step.point for u in infantry):
                    ctx.last_canvas[195:240, 180:235] = 0
                    removed.append(step.point)

    runner.executor.act = incident_camera
    runner.start('army:event-watch', 'army', event_watch(ctx, army))
    for _ in range(300):
        runner.tick(world.snapshot(), None)
        time[0] += .1
        if not runner.tasks:
            break
    assert not runner.tasks and ctx.camera == 'home' and len(removed) == 1
    event = next(i for i, s in enumerate(steps) if s.keys == ('space',))
    orders = [i for i, s in enumerate(steps) if s.kind == 'click' and s.keys == ('ctrl', 'shift')]
    assert len(orders) == army.book['defense']['session_rounds']
    assert event < min(orders) < max(orders) < len(steps) - 1
    assert not any(s.keys == ('p',) for s in steps[event:])
    assert steps[-2].keys == ('h',)
