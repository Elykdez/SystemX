"""Strategy v2: radar reading, attack waves, army postures, production rules and System One questions.

The radar fixture is the radar box of a recorded game (2026-09-23): our base at the left,
enemy buildings in blue at the right, the rest of the map shrouded.
"""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.army import Army, attack_wave, choose_target
from addons.chronodivide.executor import Trust
from addons.chronodivide.intent import StandingIntent
from addons.chronodivide.layout import Layout
from addons.chronodivide.perception.radar import read_radar
from addons.chronodivide.reflex import Reflex
from addons.chronodivide import skills
from addons.chronodivide.tasks import TaskRunner
from addons.chronodivide.world import Slot, World

FIXTURES = Path(__file__).with_name('fixtures')
LAYOUT = Layout.for_canvas(1024, 768)
BOX = LAYOUT.regions['radar']


def radar_canvas(name='radar-online.png'):
    canvas = np.zeros((768, 1024, 3), np.uint8)
    canvas[BOX.y:BOX.y + BOX.h, BOX.x:BOX.x + BOX.w] = cv2.imread(str(FIXTURES / name))
    return canvas


def online_radar():
    return read_radar(radar_canvas(), BOX)


class Recorder:
    def __init__(self):
        self.acts = []

    def act(self, name, lane, steps, commands=1):
        self.acts.append((name, lane, [(s.kind, s.point, s.keys, s.button) for s in steps]))


def setup(executor=None):
    time = [1000.0]
    world = World(clock=lambda: time[0])
    world.set('mode', 'hud')
    ctx = skills.Context(LAYOUT, world)
    executor = executor or Recorder()
    runner = TaskRunner(executor, Trust(), clock=lambda: time[0])
    intent = StandingIntent()
    army = Army(ctx, runner, intent, clock=lambda: time[0])
    army.census_enabled = False          # counted in its own tests
    army.book['attack']['keep_home_infantry'] = False  # legacy wave tests have no selection-rendering game
    reflex = Reflex(ctx, runner, None, intent, clock=lambda: time[0], army=army)
    runner.on_failure = reflex.failed
    return world, ctx, runner, army, reflex, executor, time


@pytest.mark.parametrize('name', ['army:focus', 'army:census', 'army:defend', 'army:regroup'])
@pytest.mark.parametrize('sidebar', ['look:infantry', 'queue:rhino_tank'])
def test_army_observation_is_not_cleared_by_a_new_sidebar_task(monkeypatch, name, sidebar):
    from addons.chronodivide.executor import Step
    from addons.chronodivide.tasks import Act, Look, Poll
    from addons.chronodivide.perception.unitboxes import our_units
    from addons.chronodivide.tests.test_unitboxes import draw_bar
    world, ctx, runner, army, reflex, _, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    observed = []

    class SelectionGame(Recorder):
        def act(self, task, lane, steps, commands=1):
            super().act(task, lane, steps, commands)
            ctx.last_canvas[:] = 0
            if lane == 'army':
                draw_bar(ctx.last_canvas, 400, 300, 36)

    def read_army():
        yield Act([Step.key('1')])
        yield Poll(lambda *_: False, .3)
        observed.append((yield Look(lambda *_: len(our_units(ctx.last_canvas)))))

    def use_sidebar():
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key('e')])

    runner.executor = SelectionGame()
    monkeypatch.setattr(reflex, 'plan', lambda *_: (sidebar, use_sidebar, 'infantry'))
    runner.start(name, 'army', read_army())
    runner.tick(world.snapshot(), None)
    for _ in range(10):
        time[0] += .1
        reflex.step(world.snapshot())
        runner.tick(world.snapshot(), None)
        if observed:
            break
    assert observed == [1]
    reflex.step(world.snapshot())
    assert runner.tasks['sidebar'].name == sidebar  # Sidebar input resumes after the observation.


@pytest.mark.parametrize('failure,tanks,flak,siege,threat,cash,offered', [
    (1, 6, 3, 0, False, 800, True),
    (0, 6, 3, 0, False, 1800, False),
    (1, 3, 3, 0, False, 1800, False),
    (1, 6, 1, 0, False, 1800, False),
    (1, 6, 3, 2, False, 1800, True),
    (1, 6, 3, 4, False, 1800, False),
    (1, 24, 6, 4, False, 1800, True),
    (1, 24, 6, 6, False, 1800, False),
    (1, 6, 3, 0, True, 1800, False),
    (1, 6, 3, 0, False, 799, False),
])
def test_siege_support_requires_failed_attack_cover_budget_and_quiet_base(
        failure, tanks, flak, siege, threat, cash, offered):
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(ore_refinery=2, war_factory=1, radar_tower=1)
    world.units_built.update(rhino_tank=tanks, flak_track=flak, war_miner=2, v3_launcher=siege)
    army.failures = failure
    army.threat_at = time[0] if threat else -1e9
    available = ['rhino_tank', 'flak_track', 'v3_launcher']
    options, needs = reflex.unit_options('vehicles', available, cash, time[0])
    assert ('v3_launcher' in options) == offered
    assert 'v3_launcher' not in reflex.unit_options('vehicles', ['rhino_tank'], cash, time[0])[0]


def test_rich_queue_does_not_exceed_four_siege_launchers():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.set('credits', 6000)
    world.units_built['v3_launcher'] = 3
    assert reflex.queue_count('v3_launcher') == 1


def test_rebuilt_rhino_production_allows_replacement_siege_launchers():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.set('credits', 6000)
    world.units_built.update(rhino_tank=24, v3_launcher=4)
    assert reflex.queue_count('v3_launcher') == 2
    world.units_built['v3_launcher'] = 5
    assert reflex.queue_count('v3_launcher') == 1


def run(runner, world, time, until, ticks=30, step=0.1):
    for _ in range(ticks):
        runner.tick(world.snapshot(), None)
        if until():
            return True
        time[0] += step
    return until()


# -- radar perception ---------------------------------------------------------

def test_online_radar_shows_map_outline_camera_box_our_base_and_the_enemy():
    radar = online_radar()
    assert radar.online
    assert radar.bounds == (886, 48, 991, 157)
    assert radar.view == (913, 80, 930, 108)
    area, home = radar.own[0]
    assert area >= 30 and abs(home[0] - 914) <= 3 and abs(home[1] - 92) <= 3
    area, enemy = radar.enemy[0]
    assert area >= 20 and abs(enemy[0] - 971) <= 3 and abs(enemy[1] - 108) <= 3
    assert 0.05 < radar.explored_share < 0.5


def test_emblem_and_static_are_offline_radar():
    for name in ('frame-play.png', 'base-reactors.png', 'enemy-near-base.png', 'frame-vehicles-queued.png'):
        assert not read_radar(cv2.imread(str(FIXTURES / name)), BOX).online, name
    assert not read_radar(radar_canvas('radar-starting.png'), BOX).online


def test_compressed_radar_keeps_outline_camera_and_enemy_target():
    for quality in (80, 90, 95):
        encoded = cv2.imencode('.jpg', radar_canvas(), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        radar = read_radar(cv2.imdecode(encoded, cv2.IMREAD_COLOR), BOX, compressed=True)
        assert radar.online and radar.bounds == (886, 48, 991, 157)
        assert radar.view == (913, 80, 930, 108)
        area, home = radar.own[0]
        assert area >= 30 and abs(home[0] - 914) <= 3 and abs(home[1] - 92) <= 3
        area, enemy = radar.enemy[0]
        assert area >= 20 and abs(enemy[0] - 971) <= 3 and abs(enemy[1] - 108) <= 3
        assert choose_target(radar, home) == (enemy, 'enemy')


def test_compressed_recorded_radars_keep_contacts_and_camera_geometry():
    for name in ('army-selected.png', 'army-battle.png', 'enemy-base-siege.png', 'cursor-radar.png'):
        canvas = cv2.imread(str(FIXTURES / name))
        native = read_radar(canvas, BOX)
        encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
        radar = read_radar(cv2.imdecode(encoded, cv2.IMREAD_COLOR), BOX, compressed=True)
        assert radar.online and radar.bounds == native.bounds and radar.view == native.view, name
        assert bool(radar.own) == bool(native.own) and bool(radar.enemy) == bool(native.enemy), name
        if native.enemy:
            assert abs(radar.enemy[0][1][0] - native.enemy[0][1][0]) <= 3, name
            assert abs(radar.enemy[0][1][1] - native.enemy[0][1][1]) <= 3, name


def test_compressed_emblem_and_static_do_not_become_online():
    canvases = [cv2.imread(str(FIXTURES / name)) for name in
                ('frame-play.png', 'base-reactors.png', 'enemy-near-base.png', 'frame-vehicles-queued.png')]
    canvases.append(radar_canvas('radar-starting.png'))
    for canvas in canvases:
        encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
        assert not read_radar(cv2.imdecode(encoded, cv2.IMREAD_COLOR), BOX, compressed=True).online


def test_mirror_guess_lands_near_the_real_enemy_base():
    radar = online_radar()
    home = radar.own[0][1]
    guess = radar.guesses(home)[0]
    enemy = radar.enemy[0][1]
    assert abs(guess[0] - enemy[0]) <= 12 and abs(guess[1] - enemy[1]) <= 12


def test_camera_box_maps_radar_points_into_the_tactical_view():
    radar = online_radar()
    x0, y0, x1, y1 = radar.view
    tactical = LAYOUT.tactical
    x, y = radar.to_tactical(((x0 + x1) / 2, (y0 + y1) / 2), tactical)
    assert abs(x - tactical.w / 2) <= 30 and abs(y - tactical.h / 2) <= 30
    left, top = radar.to_tactical((x0, y0), tactical)
    assert left < 40 and top < 20


# -- targets ----------------------------------------------------------------------

def test_seen_enemy_objects_are_the_target():
    radar = online_radar()
    point, kind = choose_target(radar, radar.own[0][1])
    assert kind == 'enemy' and point == radar.enemy[0][1]


def test_without_sightings_the_mirror_guess_is_tried_until_explored_then_the_next():
    radar = online_radar()
    home = radar.own[0][1]
    hidden = replace(radar, enemy=(), explored=np.zeros_like(radar.explored))
    point, kind = choose_target(hidden, home)
    assert kind == 'guess' and point == hidden.guesses(home)[0]
    assert choose_target(hidden, home, current=point) == (point, 'guess')          # still shrouded: keep it
    seen = hidden.explored.copy()
    x, y = point[0] - radar.origin[0], point[1] - radar.origin[1]
    seen[y - 8:y + 9, x - 8:x + 9] = True                                           # explored, and empty
    after = replace(hidden, explored=seen)
    point2, kind2 = choose_target(after, home, current=point, cleared=[point])
    assert kind2 == 'guess' and point2 == hidden.guesses(home)[1]


def test_no_radar_no_target():
    assert choose_target(None, (900, 90)) == (None, None)
    radar = online_radar()
    assert choose_target(replace(radar, online=False), (900, 90)) == (None, None)


# -- attack waves -------------------------------------------------------------------

class CameraGame(Recorder):
    """A radar click moves the camera: the radar reading's camera box follows the click."""

    def __init__(self, world, follows=True):
        super().__init__()
        self.world, self.follows = world, follows

    def act(self, name, lane, steps, commands=1):
        super().act(name, lane, steps, commands)
        radar = self.world.radar.value
        for step in steps:
            if step.kind == 'click' and step.button == 'left' and BOX.contains(step.point) and self.follows \
                    and not step.keys:
                x, y = step.point
                self.world.set('radar', replace(radar, view=(x - 8, y - 14, x + 9, y + 14)))
            if step.kind == 'key' and step.keys == ('h',):
                self.world.set('radar', replace(radar, view=(913, 80, 930, 108)))


def test_attack_wave_moves_the_camera_by_radar_then_attack_moves_the_whole_army_and_returns_home():
    world, ctx, runner, army, reflex, executor, time = setup()
    executor = runner.executor = CameraGame(world)
    world.set('radar', online_radar())
    target = world.radar.value.enemy[0][1]
    runner.start('army:attack', 'army', attack_wave(ctx, target, 'enemy'))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert runner.finished[-1].status == 'done'
    first, order, back = [a[2] for a in executor.acts]
    assert first[0][:2] == ('click', ctx.clear_point) and first[0][3] == 'right'     # clear the selection
    assert first[1][:2] == ('click', target) and first[1][3] == 'left'               # camera to the target
    assert first[2][:2] == ('move', ctx.radar_park)                                   # the cursor stays on the radar
    keys = [step[2] for step in order if step[0] == 'key']
    click = next(step for step in order if step[0] == 'click')
    assert keys == [('p',), ('p',)] and click[2] == ('ctrl', 'shift')              # P twice: the whole map
    assert LAYOUT.tactical.contains(click[1])                                         # aimed in the view
    assert back[0][3] == 'right' and back[1][2] == ('h',)                             # deselect, camera home
    assert ctx.camera == 'home'
    assert any(e['kind'] == 'order' and e.get('camera') for e in world.events)


def test_when_the_camera_does_not_follow_the_order_is_given_on_the_radar():
    world, ctx, runner, army, reflex, executor, time = setup()
    executor = runner.executor = CameraGame(world, follows=False)
    world.set('radar', online_radar())
    target = world.radar.value.enemy[0][1]
    runner.start('army:attack', 'army', attack_wave(ctx, target, 'enemy'))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    click = next(step for step in executor.acts[1][2] if step[0] == 'click')
    assert click[:3] == ('click', target, ('ctrl', 'shift'))


def attack_setup(units=8):
    world, ctx, runner, army, reflex, executor, time = setup()
    world.set('radar', online_radar())
    ctx.radar_home = world.radar.value.own[0][1]
    world.units_built['rhino_tank'] = units
    army.intent.set('posture', 'attack', 'test')
    return world, ctx, runner, army, reflex, executor, time


def test_attack_posture_sends_waves_for_new_units_and_on_a_timer(monkeypatch):
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    # This test covers waves, without inter-wave camera sessions.
    monkeypatch.setitem(army.book['battle'], 'every', 1e12)
    plan = army.step(world.snapshot())
    assert plan[0] == 'army:attack' and runner.busy('army') and army.campaign.active
    assert army.campaign.waves == 0 and army.campaign.kind == 'enemy'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.waves == 1
    time[0] += 9
    army.step(world.snapshot())
    assert army.campaign.waves == 1                                   # nothing new: no wave yet
    world.units_built['rhino_tank'] += 3
    time[0] += 1.5
    army.step(world.snapshot())
    assert army.campaign.waves == 1                                   # three new tanks wait for a fourth
    world.units_built['rhino_tank'] += 1
    time[0] += 1.5
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.waves == 2                                   # four new tanks join together
    time[0] += 41
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.waves == 3                                   # survivors re-targeted


def test_gather_ends_the_campaign_and_the_next_attack_needs_new_units():
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=9)
    army.step(world.snapshot())
    runner.cancel_all('test')
    army.intent.set('posture', 'gather', 'test')
    time[0] += 2
    army.step(world.snapshot())
    assert not army.campaign.active and army.available() == 0
    assert 'attack' not in reflex.feasible_postures(time[0])
    world.units_built['rhino_tank'] += 12                          # enough new units to attack again
    assert 'attack' in reflex.feasible_postures(time[0])


def test_army_orders_wait_while_a_building_is_being_placed():
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    runner.start('place:barracks', 'sidebar', iter(()))
    assert army.step(world.snapshot())[0] == 'army:attack' and not runner.busy('army')
    runner.cancel_all('test')
    time[0] += 1.5
    army.step(world.snapshot())
    assert runner.busy('army')
    world.structures['construction_yard'] = 1
    world.update_sidebar('buildings', [Slot(0, 'barracks', 'ready', 1.0)])
    assert reflex.step(world.snapshot())[0] == 'place:barracks' and not runner.busy('sidebar')


def test_defend_posture_hunts_enemies_seen_near_home_on_the_radar():
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    raid = ((12, (home[0] + 6, home[1] + 4)),)
    world.set('radar', replace(radar, enemy=raid))
    ctx.radar_home = home
    world.units_built['rhino_tank'] = 3
    army.intent.set('posture', 'defend', 'test')
    plan = army.step(world.snapshot())
    assert plan == ('army:defend-radar', raid[0][1]) and runner.busy('army')
    assert any(e['kind'] == 'attack' for e in world.events)
    assert 'defend' in reflex.feasible_postures(time[0])


# -- strategy decisions ---------------------------------------------------------------

class Broker:
    def __init__(self, time):
        self.time, self.submitted = time, []

    def clock(self):
        return self.time[0]

    def submit(self, decision):
        self.submitted.append(decision)

    def busy_keys(self):
        return set()


def test_strategy_asks_objective_and_posture_with_facts_in_the_criteria_words():
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=9)
    army.intent.set('posture', 'gather', 'test')
    army.intent.set('attack_size', 9, 'systemone')
    world.structures.update(construction_yard=1, ore_refinery=2, war_factory=1)
    world.units_built['war_miner'] = 2
    world.set('credits', 7000)
    army.census_tanks, army.census_at, army.census_healthy_vehicles = 12, time[0], 12
    from addons.chronodivide.intel import Intel
    army.intel = Intel(lambda: time[0])
    army.intel.enemy_base = (970, 108)
    army.intel.record_look('enemy_base', (970, 108),
                         dict(enemy_objects=2, enemy_pixels=200, own_objects=0, shroud=0.1), time[0])
    reflex.broker = Broker(time)
    reflex.ask_strategy(time[0])
    asked = {next(iter(d.questions)): d for d in reflex.broker.submitted}    # one focused request per question
    assert set(asked) == {'objective', 'focus', 'attack_size'}
    objective, focus = asked['objective'], asked['focus']
    assert (set(objective.questions), set(focus.questions)) == ({'objective'}, {'focus'})
    assert set(objective.state) == {'economy', 'money', 'army', 'base_threat'}
    assert set(focus.state) == {'army', 'attack', 'base_threat'}
    assert objective.state['economy'] == 'enough miners and refineries'
    assert 'credits pile up unused' in objective.state['money']
    assert 'credits pile up unused' in objective.questions['objective']['criteria']['tech_up']
    assert set(focus.questions['focus']['criteria']) == {'build', 'attack'}
    assert 'attack size reached' in focus.state['army'] and focus.default['focus']['choice'] == 'attack'
    objective.on_result(objective, {'objective': {'choice': 'tech_up'}}, 'systemone', {})
    focus.on_result(focus, {'focus': {'choice': 'attack'}}, 'systemone', {})
    assert (army.intent.objective, army.intent.posture) == ('tech_up', 'attack')


def test_systemone_decides_how_many_tanks_the_attack_needs():
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=0)
    medium = reflex.book.option('attack_size', 'medium')['tanks']
    large = reflex.book.option('attack_size', 'large')['tanks']
    world.units_built['rhino_tank'] = medium - 1
    army.intent.set('posture', 'gather', 'test')
    reflex.broker = Broker(time)
    reflex.ask_strategy(time[0])
    size = next(d for d in reflex.broker.submitted if 'attack_size' in d.questions)
    assert size.state == {'game_time': 'early game (0 minutes played)',
                          'failed_attacks': 'none: no attack failed yet',
                          'enemy_defenses': 'unknown (the enemy base was not looked at yet)',
                          'reinforcements': reflex.book.say('attack_size', 'reinforcements', 'not_limited')}
    criteria = size.questions['attack_size']['criteria']
    assert list(criteria) == ['medium', 'large', 'larger', 'largest'] and 'unknown' in criteria['large']
    assert size.default['attack_size']['choice'] == 'large' and reflex.attack_size(time[0]) == large
    assert 'attack' not in reflex.feasible_postures(time[0])
    size.on_result(size, {'attack_size': {'choice': 'medium'}}, 'systemone', {})
    assert army.intent.attack_size == medium and reflex.attack_size(time[0]) == medium
    assert 'attack' not in reflex.feasible_postures(time[0])                  # one short of System One's size
    world.units_built['rhino_tank'] += 1
    assert 'attack' in reflex.feasible_postures(time[0])
    army.campaign.active, army.campaign.started = True, time[0]
    reflex.broker.submitted.clear()
    time[0] += 16
    reflex.ask_strategy(time[0])
    assert not any('attack_size' in d.questions for d in reflex.broker.submitted)  # not during an attack


def test_late_game_only_attacks_with_a_smaller_army_after_observing_weak_defenses():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    world.set('radar', online_radar())
    small = reflex.book.option('attack_size', 'small')['tanks']
    world.units_built['rhino_tank'] = small
    assert 'attack' not in reflex.feasible_postures(time[0])
    world.started = time[0] - 13 * 60
    assert reflex.attack_size(time[0]) == reflex.book.option('attack_size', 'large')['tanks']
    assert 'small' not in reflex.feasible_attack_sizes(time[0])
    assert 'attack' not in reflex.feasible_postures(time[0])
    army.intel.strength = 'weak'
    assert 'attack' in reflex.feasible_postures(time[0]) and reflex.attack_size(time[0]) == small


def test_late_size_rules_keep_unknown_medium_strong_and_failed_attacks_conservative():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    world.started = time[0] - 13 * 60
    for strength, failures, option in [(None, 0, 'large'), ('medium', 0, 'large'), ('strong', 0, 'larger'),
                                       ('weak', 0, 'small'), ('weak', 1, 'larger'), ('weak', 2, 'largest')]:
        army.intel.strength, army.failures = strength, failures
        assert reflex.attack_size(time[0]) == reflex.book.option('attack_size', option)['tanks']
        if option != 'small':
            assert 'small' not in reflex.feasible_attack_sizes(time[0])
    assert 'weakened' not in reflex.attack_size_facts(time[0])[0]['game_time']


def test_small_size_broker_answer_is_rejected_when_the_enemy_evidence_changes():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    world.started = time[0] - 13 * 60
    army.intel.strength = 'weak'
    reflex.broker = Broker(time)
    reflex.ask_strategy(time[0])
    decision = next(d for d in reflex.broker.submitted if 'attack_size' in d.questions)
    assert 'small' in decision.questions['attack_size']['criteria']
    army.intel.strength = 'strong'
    decision.on_result(decision, {'attack_size': {'choice': 'small'}}, 'systemone', {})
    larger = reflex.book.option('attack_size', 'larger')['tanks']
    assert army.intent.attack_size is None and reflex.attack_size(time[0]) == larger
    army.intent.set('attack_size', reflex.book.option('attack_size', 'small')['tanks'], 'systemone')
    assert reflex.attack_size(time[0]) == larger  # a previously accepted answer also expires
    time[0] += 16
    reflex.broker.submitted.clear()
    reflex.ask_strategy(time[0])
    decision = next(d for d in reflex.broker.submitted if 'attack_size' in d.questions)
    assert set(decision.questions['attack_size']['criteria']) == {'larger', 'largest'}


def test_current_base_raid_forces_defense_and_rejects_delayed_attack_answer():
    world, ctx, runner, army, reflex, executor, time = setup()
    reflex.broker = Broker(time)
    reflex.ask_strategy(time[0])
    time[0] += 4
    army.threat_at, army.seen = time[0], ((300, 300), 200)
    reflex.ask_strategy(time[0])                       # well before the 15 s interval
    decision = reflex.broker.submitted[-1]
    assert decision.kind == 'urgent' and army.intent.posture == 'defend'
    assert reflex.feasible_postures(time[0]) == ['defend']
    reflex._strategy_result(decision, {'posture': {'choice': 'attack'}}, 'systemone', {})
    assert army.intent.posture == 'defend'
    time[0] += 30
    assert 'attack' not in reflex.feasible_postures(time[0])  # still lacks army/radar
    reflex.ask_strategy(time[0])
    assert army.intent.posture == 'gather'


def test_strong_defenses_reject_small_stored_attack_size_even_early():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    army.intel.strength = 'strong'
    larger = reflex.book.option('attack_size', 'larger')['tanks']
    army.intent.set('attack_size', larger - 1, 'systemone')
    assert reflex.attack_size(time[0]) == larger
    assert reflex.feasible_attack_sizes(time[0]) == ['larger', 'largest']


@pytest.mark.parametrize('faction', ['allied', 'soviet'])
@pytest.mark.parametrize('stall,census_age,shortfall,cash,failures,option', [
    (90, 0, 0, 800, 0, 'large'),
    (89, 0, 0, 800, 0, 'larger'),
    (90, 60, 0, 800, 0, 'larger'),
    (90, 0, 1, 800, 0, 'larger'),
    (90, 0, 0, 4000, 0, 'larger'),
    (90, 0, 0, 800, 1, 'larger'),
    (90, 0, 0, 800, 2, 'largest'),
])
def test_income_stall_allows_a_confirmed_large_army_without_weakening_failure_rules(
        faction, stall, census_age, shortfall, cash, failures, option):
    from addons.chronodivide import playbook
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    book = playbook.load(Path(__file__).resolve().parents[1] / 'profiles' / f'startlux-ra2-{faction}.json')
    army.book = reflex.book = book
    world.knowledge = playbook.knowledge(book)
    army.intel.strength, army.failures = 'strong', failures
    world.set('credits', cash)
    army.miners.funding_at, army.miners.income_at = time[0] - stall, time[0]
    army.census_at = time[0] - census_age
    army.census_tanks = book.option('attack_size', 'large')['tanks'] - shortfall
    if faction == 'allied' and failures == 0:
        option = 'large'  # A first ranged siege need not mass twenty vehicles for static defenses.
    assert reflex.attack_size(time[0]) == book.option('attack_size', option)['tanks']
    assert 'small' not in reflex.feasible_attack_sizes(time[0])
    assert 'medium' not in reflex.feasible_attack_sizes(time[0])
    if option == 'large':
        reflex.broker = Broker(time)
        reflex.ask_strategy(time[0])
        decision = next(d for d in reflex.broker.submitted if 'attack_size' in d.questions)
        if stall >= 90 and census_age < 60 and shortfall == 0 and cash < 4000:
            assert 'Waiting for more vehicles' in decision.state['reinforcements']
        decision.on_result(decision, {'attack_size': {'choice': 'large'}}, 'systemone', {})
        assert army.intent.attack_size == book.option('attack_size', 'large')['tanks']
        army.miners.funding_at = time[0]  # A delivery/progress resumes the conservative threshold.
        expected = 'large' if faction == 'allied' else 'larger'
        assert reflex.attack_size(time[0]) == book.option('attack_size', expected)['tanks']


def test_infeasible_posture_falls_back_to_the_rule_at_once():
    world, ctx, runner, army, reflex, executor, time = setup()
    army.intent.set('posture', 'defend', 'systemone')
    reflex.ask_strategy(time[0])                       # no threat was ever seen
    assert army.intent.posture == 'gather' and army.intent.view()['provenance']['posture'] == 'rule'


# -- production -----------------------------------------------------------------------------

def test_vehicles_rules_first_tanks_then_miners_then_tanks_with_some_flak():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(ore_refinery=1, war_factory=1)
    world.set('credits', 5000)
    available = ('war_miner', 'rhino_tank', 'flak_track', 'terror_drone')
    anti_air, miners = reflex.book['units']['anti_air'], reflex.book['units']['miners']
    world.units_built['rhino_tank'] = anti_air['after_tanks'] - 1
    options, needs = reflex.unit_options('vehicles', available, 5000, time[0])
    assert options == ['rhino_tank']                                 # the early rush comes first
    world.units_built['rhino_tank'] = anti_air['after_tanks']
    options, needs = reflex.unit_options('vehicles', available, 5000, time[0])
    assert options == ['flak_track']                                 # air cover starts at its configured gate
    tanks = max(anti_air['after_tanks'], miners['after_tanks'])
    world.units_built['rhino_tank'] = tanks
    options, needs = reflex.unit_options('vehicles', available, 5000, time[0])
    assert options == ['flak_track']                               # air cover before expansion
    world.units_built['flak_track'] = 1
    options, needs = reflex.unit_options('vehicles', available, 5000, time[0])
    assert options[0] == 'war_miner' and 'too few miners' in needs[0]
    flak_needed = (tanks + anti_air['one_per_tanks'] - 1) // anti_air['one_per_tanks']
    world.units_built.update(war_miner=miners['per_refinery'] - 1, flak_track=flak_needed)
    options, needs = reflex.unit_options('vehicles', available, 5000, time[0])
    assert options == ['rhino_tank']                                 # two miners for the one refinery
    world.units_built['rhino_tank'] += anti_air['one_per_tanks']
    options, needs = reflex.unit_options('vehicles', available, 5000, time[0])
    assert options == ['flak_track', 'rhino_tank'] and any('anti-air' in n for n in needs)
    assert reflex.next_item('vehicles', available, time[0]) == 'flak_track'      # rule default, no System One
    assert 'terror_drone' not in options


def test_infantry_stops_after_the_early_conscripts_unless_credits_pile_up():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.set('credits', 1200)
    available = ('conscript', 'engineer', 'attack_dog', 'tesla_trooper')
    assert reflex.next_item('infantry', available, time[0]) == 'attack_dog'        # two dog scouts first
    world.units_built['attack_dog'] = 2
    world.units_built['attack_dog'] = 4
    assert reflex.next_item('infantry', available, time[0]) == 'conscript'
    world.units_built['conscript'] = 6
    assert reflex.next_item('infantry', available, time[0]) is None             # money goes to tanks
    world.set('credits', 4000)
    assert reflex.next_item('infantry', available, time[0]) is None             # no war factory yet: tanks first
    world.structures['war_factory'] = 1
    assert reflex.next_item('infantry', available, time[0]) is None             # nor before eight tanks
    world.units_built['rhino_tank'] = 8
    assert reflex.next_item('infantry', available, time[0]) is None             # no proven advantage


def test_unverified_unit_queue_counts_the_unit_and_holds_the_lane():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, war_factory=1, ore_refinery=1)
    world.set('credits', 5000)
    world.update_sidebar('vehicles', [Slot(0, 'war_miner', 'available'), Slot(1, 'rhino_tank', 'available')])
    assert reflex.step(world.snapshot())[0] == 'queue:rhino_tank'
    assert run(runner, world, time, lambda: not runner.busy('sidebar'), ticks=40)
    assert runner.finished[-1].status == 'failed' and 'expectation' in runner.finished[-1].reason
    assert world.units_built['rhino_tank'] == 1                    # counted: no second click at once
    world.update_sidebar('vehicles', [Slot(0, 'war_miner', 'available'), Slot(1, 'rhino_tank', 'available')])
    plan = reflex.step(world.snapshot(), act=False)
    assert plan is None or not plan[0].startswith('queue:')         # the lane settles first
    time[0] += 9
    world.update_sidebar('vehicles', [Slot(0, 'war_miner', 'available'), Slot(1, 'rhino_tank', 'available')])
    assert reflex.step(world.snapshot(), act=False)[0] == 'queue:rhino_tank'


def test_production_question_offers_only_sensible_units_with_when_to_choose_criteria():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(ore_refinery=2, war_factory=1)
    world.units_built.update(rhino_tank=8, flak_track=1)
    world.set('credits', 6000)
    reflex.broker = Broker(time)
    available = ('war_miner', 'rhino_tank', 'flak_track', 'terror_drone', 'apocalypse_tank', 'mcv')
    assert reflex.next_item('vehicles', available, time[0]) is None             # asked System One
    decision = reflex.broker.submitted[-1]
    criteria = decision.questions['production']['criteria']
    assert list(criteria) == ['war_miner', 'flak_track', 'rhino_tank']
    assert 'too few miners' in criteria['war_miner'] and 'default tank' in criteria['rhino_tank']
    assert any('too few miners' in need for need in decision.state['needs'])
    decision.on_result(decision, {'production': {'choice': 'flak_track'}}, 'systemone', {})
    assert reflex.next_item('vehicles', available, time[0]) == 'flak_track'


# -- scouting and looks (intel) ---------------------------------------------------------------

from addons.chronodivide.intel import Intel, survey


def intel_setup(executor=None):
    world, ctx, runner, army, reflex, executor, time = setup(executor)
    army.intel = Intel(lambda: time[0])
    return world, ctx, runner, army, reflex, executor, time


def test_failed_local_route_does_not_keep_offering_an_unexecutable_explore_request():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    army.scouts_ready = True
    assert 'explore' in reflex.feasible_intel(time[0])
    army.explore_failed = True
    assert 'explore' not in reflex.feasible_intel(time[0])


def test_survey_counts_enemy_and_own_objects_in_view():
    tactical = LAYOUT.tactical
    enemy = survey(cv2.imread(str(FIXTURES / 'enemy-near-base.png')), tactical)
    assert enemy['enemy_objects'] >= 3 and enemy['own_objects'] >= 3 and enemy['shroud'] < 0.5
    quiet = survey(cv2.imread(str(FIXTURES / 'base-reactors.png')), tactical)
    assert quiet['enemy_objects'] == 0 and quiet['own_objects'] >= 1 and quiet['shroud'] > 0.3


def test_enemy_base_is_found_on_the_radar_and_its_look_sets_the_attack_size():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    radar = online_radar()
    home = radar.own[0][1]
    intel = army.intel
    intel.update(replace(radar, enemy=((12, (home[0] + 6, home[1] + 4)),)), home, time[0])
    assert intel.enemy_base is None                                   # a raid near home is not their base
    intel.update(radar, home, time[0])
    assert intel.enemy_base == radar.enemy[0][1]
    assert reflex.intel_facts(time[0])['enemy_base'] == 'found; not looked at recently'
    intel.record_look('enemy_base', intel.enemy_base, dict(enemy_objects=14, enemy_pixels=2500, own_objects=0,
                                                           shroud=0.1), time[0])
    assert intel.strength == 'strong'
    assert reflex.intel_facts(time[0] + 5)['enemy_base'] == 'found; looked at recently (5 s ago, strong defenses)'
    assert reflex.attack_size_facts(time[0])[0]['enemy_defenses'] == 'strong'
    assert reflex.attack_size(time[0]) == reflex.book.option('attack_size', 'larger')['tanks']


def test_scout_request_sends_only_the_verified_team_with_a_plain_move():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    executor = runner.executor = CameraGame(world)
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, enemy=(), explored=np.zeros_like(radar.explored)))
    ctx.radar_home = home
    world.units_built['rhino_tank'] = 12
    army.scouts_ready = army.explored = True
    assert set(reflex.feasible_intel(time[0])) == {'scout', 'nothing'}
    army.intent.set('intel', 'scout', 'systemone')
    plan = army.step(world.snapshot())
    assert plan[0] == 'army:scout' and army.intel.scout_out(time[0]) and army.intent.intel is None
    assert run(runner, world, time, lambda: not runner.busy('army'))
    team = reflex.book['intel']['scout_team']
    order = next(a[2] for a in executor.acts if a[2][0][2] == (team,))
    click = next(step for step in order if step[0] == 'click')
    assert click[2] == () and click[3] == 'left'                                      # a move, not attack-move
    assert not any(step[2] == ('p',) for act in executor.acts for step in act[2])
    assert 'scout' not in reflex.feasible_intel(time[0])                               # already on the way
    world.set('radar', radar)                                                            # the scout found their base
    time[0] += 2
    assert army.step(world.snapshot())[0] == 'army:return' and army.intel.enemy_base is not None
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert any(e['kind'] == 'order' and e['order'] == 'return_home' for e in world.events)
    assert not any(step[2] == ('p',) for act in executor.acts for step in act[2])
    returned = next(a[2] for a in executor.acts if a[0] == 'army:return' and a[2][0][2] == (team,))
    assert any(step[0] == 'click' and step[3] == 'left' for step in returned)


def test_scout_request_without_a_verified_team_never_selects_the_combat_army():
    for tanks in (2, 12):
        world, ctx, runner, army, reflex, executor, time = intel_setup()
        radar = online_radar()
        world.set('radar', replace(radar, enemy=(), explored=np.zeros_like(radar.explored)))
        ctx.radar_home = radar.own[0][1]
        world.units_built['rhino_tank'] = tanks
        campaign = army.campaign
        campaign.active = tanks > 2
        campaign.target, campaign.waves = (960, 120), 3
        army.intent.set('intel', 'scout', 'systemone')
        assert not army.scouts_ready
        assert 'scout' not in reflex.feasible_intel(time[0])
        assert army._intel(time[0], True) is None
        assert army.intent.intel is None and army.intel.scout is None
        assert not runner.tasks and executor.acts == []
        assert army.campaign is campaign and campaign.active == (tanks > 2)
        assert campaign.target == (960, 120) and campaign.waves == 3


def test_unavailable_scout_team_does_not_recall_the_combat_army():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    army.intel.send_scout((960, 120), time[0])
    army.intel.scout['found'] = True
    assert not army.scouts_ready
    assert army._intel(time[0], True) is None
    assert not runner.tasks and executor.acts == []
    assert not army.intel.scout.get('returned')


def test_look_moves_only_the_camera_and_records_what_it_saw():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    executor = runner.executor = CameraGame(world)
    radar = online_radar()
    world.set('radar', radar)
    ctx.radar_home = radar.own[0][1]
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-near-base.png'))
    army.intel.update(radar, ctx.radar_home, time[0])
    assert 'look_enemy_base' in reflex.feasible_intel(time[0])
    army.intent.set('intel', 'look_enemy_base', 'systemone')
    assert army.step(world.snapshot())[0] == 'army:look_enemy_base'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    keys = [step[2] for act in executor.acts for step in act[2] if step[0] == 'key']
    assert ('p',) not in keys and ('h',) in keys                                        # no order; camera home
    look = army.intel.looks['enemy_base']
    assert look['enemy_objects'] >= 3 and army.intel.strength in ('weak', 'medium')
    assert 'look_enemy_base' not in reflex.feasible_intel(time[0] + 5)                 # looked at recently


def test_intel_question_uses_its_own_worded_state():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    world.set('radar', replace(online_radar(), enemy=()))
    ctx.radar_home = (914, 92)
    world.units_built['conscript'] = 2
    army.scouts_ready = army.explored = True
    reflex.broker = Broker(time)
    reflex.ask_strategy(time[0])
    decision = next(d for d in reflex.broker.submitted if 'intel' in d.questions)
    assert decision.state == {'enemy_base': 'not found yet', 'scout': 'no scout is on the way', 'attack': 'not started',
                              'surroundings': 'the ground around our base was explored'}
    assert decision.default['intel']['choice'] == 'scout'
    decision.on_result(decision, {'intel': {'choice': 'scout'}}, 'systemone', {})
    assert army.intent.intel == 'scout'


def test_interrupted_wave_leaves_the_camera_lost_and_h_brings_it_back():
    world, ctx, runner, army, reflex, executor, time = setup()
    executor = runner.executor = CameraGame(world)
    world.set('radar', online_radar())
    runner.start('army:attack', 'army', attack_wave(ctx, world.radar.value.enemy[0][1], 'enemy'))
    runner.tick(world.snapshot(), None)                  # the camera jumped to the target
    runner.cancel_all('paused')                          # physical input pauses the agent here
    assert ctx.camera == 'lost'
    assert army.step(world.snapshot())[0] == 'army:camera'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert executor.acts[-1][2][0][2] == ('h',) and ctx.camera == 'home'


def test_army_strength_counts_tank_equivalents():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.units_built.update(conscript=6, war_miner=3, engineer=1)
    assert army.army_size() == 0                                   # cheap infantry is no attack force
    world.units_built.update(rhino_tank=5, apocalypse_tank=1)
    assert army.army_size() == 7


def test_building_that_cannot_be_placed_is_cancelled_and_skipped_for_a_while():
    world, ctx, runner, army, reflex, executor, time = setup()
    runner.on_done = reflex.done
    world.structures.update(construction_yard=1, tesla_reactor=1, ore_refinery=1, barracks=1)
    world.set('credits', 5000)
    world.update_sidebar('buildings', [Slot(0, 'war_factory', 'ready', 1.0), Slot(1, 'tesla_reactor', 'disabled')])
    reflex.failures['place:war_factory'] = 6
    assert reflex.step(world.snapshot())[0] == 'cancel:war_factory'
    runner.tick(world.snapshot(), None)
    click = executor.acts[-1][2][0]
    assert click[:2] == ('click', ctx.slot_point(0)) and click[3] == 'right'          # right click: cancel
    world.update_sidebar('buildings', [Slot(0, 'war_factory', 'available'), Slot(1, 'tesla_reactor', 'available')])
    assert run(runner, world, time, lambda: not runner.busy('sidebar'))
    assert runner.finished[-1].status == 'done' and reflex.skip['war_factory'] > time[0]
    assert reflex.next_item('buildings', ['war_factory', 'tesla_reactor'], time[0]) != 'war_factory'


def test_units_are_queued_two_at_a_time_when_money_is_plentiful():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, war_factory=1, ore_refinery=2)
    world.units_built.update(war_miner=2)
    world.set('credits', 4000)
    world.update_sidebar('vehicles', [Slot(0, 'rhino_tank', 'available')])
    assert reflex.step(world.snapshot())[0] == 'queue:rhino_tank'
    runner.tick(world.snapshot(), None)
    clicks = [s for s in executor.acts[-1][2] if s[0] == 'click']
    assert len(clicks) == 2 and clicks[0][1] == clicks[1][1] == ctx.slot_point(0)
    world.update_sidebar('vehicles', [Slot(0, 'rhino_tank', 'building', 0.1, 2)])
    assert run(runner, world, time, lambda: not runner.busy('sidebar'))
    assert world.units_built['rhino_tank'] == 2 and world.ledger.reserved() == 0
    time[0] += 1
    world.set('credits', 1500)                                       # short of money: one at a time
    world.update_sidebar('vehicles', [Slot(0, 'rhino_tank', 'available')])
    acts = len(executor.acts)
    assert reflex.step(world.snapshot())[0] == 'queue:rhino_tank'
    runner.tick(world.snapshot(), None)
    assert len(executor.acts) == acts + 1 and sum(s[0] == 'click' for s in executor.acts[-1][2]) == 1


def test_expanding_the_economy_does_not_delay_barracks_and_war_factory():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, tesla_reactor=1, ore_refinery=1)
    world.set('credits', 8000)
    army.intent.set('objective', 'expand_economy', 'systemone')
    available = ['tesla_reactor', 'ore_refinery', 'barracks']
    assert reflex.next_item('buildings', available, time[0]) == 'barracks'      # build order first
    world.structures.update(barracks=1, war_factory=1)
    world.set('credits', 2000)
    available += ['war_factory', 'radar_tower']
    assert reflex.next_item('buildings', available, time[0]) is None
    world.units_built['rhino_tank'] = 2
    assert reflex.next_item('buildings', available, time[0]) == 'ore_refinery'
    world.structures['ore_refinery'] = 2
    assert reflex.next_item('buildings', available, time[0]) is None            # radar still waits for four tanks
    world.units_built['rhino_tank'] = 4
    assert reflex.next_item('buildings', available, time[0]) == 'tesla_reactor'
    world.structures['tesla_reactor'] = 2
    assert reflex.next_item('buildings', available, time[0]) == 'radar_tower'
    world.structures['radar_tower'] = 1
    assert reflex.next_item('buildings', available, time[0]) is None            # no second factory: no money
    world.set('credits', 6000)
    assert reflex.next_item('buildings', available, time[0]) == 'war_factory'  # money piles up


def test_visible_unaffordable_refinery_does_not_time_out_into_tech_spending():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, tesla_reactor=2, ore_refinery=1,
                            barracks=1, war_factory=1, radar_tower=1)
    world.units_built['rhino_tank'] = reflex.book['units']['second_refinery']['after_tanks']
    world.set('credits', 700)
    available = ['tesla_reactor', 'ore_refinery', 'barracks', 'war_factory', 'radar_tower']
    assert reflex.next_item('buildings', available, time[0]) is None
    time[0] += reflex.book['buildings']['wait_for_next'] + 1
    assert reflex.next_item('buildings', available, time[0]) is None
    world.set('credits', 2000)
    assert reflex.next_item('buildings', available, time[0]) == 'ore_refinery'


def test_a_just_placed_reactor_is_not_followed_by_another_while_the_power_bar_catches_up():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, tesla_reactor=1, ore_refinery=1, barracks=1)
    world.set('credits', 6000)
    world.set('power', dict(level='low', fill=0.15))
    available = ['tesla_reactor', 'war_factory']
    assert reflex.next_item('buildings', available, time[0]) == 'tesla_reactor'
    reflex.done(type('Task', (), {'name': 'place:tesla_reactor'})())
    world.structures['tesla_reactor'] = 2
    assert reflex.next_item('buildings', available, time[0] + 3) == 'war_factory'       # bar still catching up
    assert reflex.next_item('buildings', available, time[0] + 12) == 'tesla_reactor'    # really low: another one


@pytest.mark.parametrize('alive', [False, True])
def test_loss_verification_follows_the_team_even_when_the_destination_is_empty(alive):
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=13)
    army.campaign.active = army.campaign.team_formed = True
    army.campaign.target = world.radar.value.enemy[0][1]
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)

    class TravelingTeam(CameraGame):
        def __init__(self):
            super().__init__(world)
            self.group_keys = 0

        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            for step in steps:
                if step.kind == 'key' and step.keys == ('1',):
                    self.group_keys += 1
                    if self.group_keys >= 2 and alive:
                        for i in range(13):
                            draw_bar(ctx.last_canvas, 150 + 60 * (i // 8), 180 + 40 * (i % 8), 36, 0.9)

    runner.executor = TravelingTeam()
    runner.start('army:verify', 'army', army._verify_army())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.loss_confirmed == (not alive)
    note = next(e for e in world.events if e['kind'] == 'army_verified')
    assert note['units'] == ([13, 13] if alive else [0, 0])


@pytest.mark.parametrize('act', [False, True])
def test_radar_outage_does_not_idle_a_live_campaign_with_home_reinforcements(act):
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=20)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    time[0] += 40
    army.census_enabled = True
    army.census_tanks, army.census_at = 5, time[0]
    army.campaign.last_focus = time[0] - army.book['battle']['every'] - 1
    world.set('radar', replace(world.radar.value, online=False))
    assert army._attack(time[0], act)[0] == 'army:focus'
    assert runner.busy('army') == act
    assert army.campaign.active


def test_an_army_lost_at_the_target_ends_the_campaign_and_the_next_attack_needs_more():
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=8)
    radar = online_radar()
    home = radar.own[0][1]
    target = radar.enemy[0][1]
    army.step(world.snapshot())                                       # first wave
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.team_formed
    assert army.campaign.active and army.campaign.target == target
    arrived = replace(radar, own=((48, home), (6, (target[0] - 4, target[1]))))
    world.set('radar', arrived)
    time[0] += 30
    army.step(world.snapshot())                                       # our dots at the enemy base
    runner.cancel_all('test')
    assert army.campaign.active and army.campaign.army_seen_at == time[0]
    world.set('radar', replace(radar, own=((48, home),)))           # ... and then gone
    time[0] += 25
    army.step(world.snapshot())
    assert army.campaign.active and army.failures == 0
    assert runner.tasks['army'].name == 'army:verify'
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    runner.executor = CameraGame(world)                           # verification must reach the battle view
    assert run(runner, world, time, lambda: not runner.busy('army'))
    time[0] += 1
    army.step(world.snapshot())
    assert not army.campaign.active and army.failures == 1 and army.intent.posture == 'gather'
    assert army.intent.attack_size is None                           # System One's size is asked again
    assert reflex.attack_size_facts(time[0])[1]['failed_attacks'] == 'one'
    required = reflex.book.option('attack_size', 'larger')['tanks']
    assert reflex.attack_size(time[0]) == required and 'attack' not in reflex.feasible_postures(time[0])
    world.units_built['rhino_tank'] += required - 1
    assert 'attack' not in reflex.feasible_postures(time[0])
    world.units_built['rhino_tank'] += 1
    assert 'attack' in reflex.feasible_postures(time[0])
    world.started = time[0] - 13 * 60                                # late game: still the bigger army
    assert reflex.attack_size(time[0]) == required


def test_missing_radar_dots_do_not_discard_a_live_attack_team():
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=12)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.team_formed
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'army-missing-radar-browser.png'))
    ctx.compressed_frames = True
    radar = read_radar(ctx.last_canvas, BOX, compressed=True)
    radar = replace(radar, own=((112, (909, 97)),))  # reproduce a missing field-army reading
    world.set('radar', radar)
    time[0] += 90
    assert army._attack(time[0], True)[0] == 'army:verify'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert not army.campaign.loss_confirmed and army.failures == 0
    assert army.campaign.confirmed_front == radar.view_center
    assert not army._army_lost(time[0], radar)
    note = next(e for e in world.events if e.get('kind') == 'army_verified')
    assert min(note['units']) >= 5


@pytest.mark.parametrize('confirmed_empty', [False, True])
def test_live_verification_survives_sidebar_delay_until_its_combat_turn(confirmed_empty):
    _, _, _, army, _, _, time = attack_setup(units=12)
    radar = online_radar()
    campaign = army.campaign
    campaign.active, campaign.target = True, radar.enemy[0][1]
    campaign.started = campaign.last_wave = time[0] - 100
    campaign.army_seen_at = time[0] - 30  # health checks outlasted the normal loss suspicion interval
    campaign.verified_at = campaign.army_seen_at
    campaign.last_focus = time[0] - 60
    campaign.loss_confirmed = confirmed_empty
    assert army._army_lost(time[0], radar) == confirmed_empty
    campaign.last_focus = time[0]
    assert army._army_lost(time[0], radar)


def test_units_beside_an_enemy_building_get_a_direct_attack_on_the_nearest_one():
    # Live game 7: attack-moved Rhinos stood idle inside the enemy base (top right of this view).
    from addons.chronodivide.army import siege_target
    siege = cv2.imread(str(FIXTURES / 'enemy-base-siege.png'))
    point = siege_target(siege, LAYOUT.tactical, (740, 120))
    assert point is not None and abs(point[0] - 602) <= 15 and abs(point[1] - 109) <= 15
    for name in ('base-reactors.png', 'enemy-near-base.png', 'frame-play.png'):         # no siege at home
        assert siege_target(cv2.imread(str(FIXTURES / name)), LAYOUT.tactical, (425, 368)) is None


def test_a_siege_wave_forces_attack_on_the_enemy_building_and_asks_for_the_next_one_sooner():
    world, ctx, runner, army, reflex, executor, time = setup()
    executor = runner.executor = CameraGame(world)
    world.set('radar', online_radar())
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-base-siege.png'))
    target = world.radar.value.enemy[0][1]
    campaign = army.campaign
    runner.start('army:attack', 'army', attack_wave(ctx, target, 'enemy', campaign=campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    order = executor.acts[1][2]
    click = next(step for step in order if step[0] == 'click')
    assert click[2] == ('ctrl',) and click[3] == 'left' and campaign.sieging


def test_radar_waits_for_the_first_tanks_so_the_rush_meets_rhinos():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, tesla_reactor=2, ore_refinery=2, barracks=1, war_factory=1)
    world.set('credits', 5000)
    available = ['tesla_reactor', 'radar_tower', 'ore_refinery']
    assert reflex.next_item('buildings', available, time[0]) is None
    world.units_built['rhino_tank'] = 4
    assert reflex.next_item('buildings', available, time[0]) == 'radar_tower'


def test_slow_opening_does_not_spend_tank_money_on_timed_expansion():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, tesla_reactor=2, ore_refinery=1, barracks=1,
                            war_factory=1, radar_tower=1)
    world.units_built['rhino_tank'] = reflex.book['units']['miners']['after_tanks'] - 1
    time[0] += 800                         # slow game or time spent paused, not more defenders
    world.set('credits', 7000)
    assert not reflex.saving_for_refinery(1500, time[0])
    assert reflex.unit_options('vehicles', ('war_miner', 'rhino_tank'), 1500, time[0])[0] == ['rhino_tank']
    available = ['ore_refinery', 'tesla_reactor', 'radar_tower']
    assert reflex.next_item('buildings', available, time[0]) is None
    world.structures['ore_refinery'] = 2
    assert reflex.next_item('buildings', available, time[0]) is None


def test_roaming_enemy_units_far_from_the_army_do_not_divert_the_attack_from_the_base():
    radar = online_radar()
    home = radar.own[0][1]
    hidden = replace(radar, explored=np.zeros_like(radar.explored))
    roaming = replace(hidden, enemy=((2, (945, 125)),))                      # a couple of enemy units mid-map
    point, kind = choose_target(roaming, home)
    assert kind == 'guess'                                                   # head for the likely base
    beside = replace(hidden, own=((48, home), (6, (942, 122))), enemy=((2, (945, 125)),))
    assert choose_target(beside, home) == ((945, 125), 'enemy')             # right beside our army: fight
    base = replace(hidden, enemy=((2, (945, 125)), (30, (971, 108))))
    assert choose_target(base, home) == ((971, 108), 'enemy')               # buildings seen: go there


def test_a_roaming_enemy_unit_is_not_remembered_as_the_enemy_base():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    radar = online_radar()
    home = radar.own[0][1]
    army.intel.update(replace(radar, enemy=((2, (945, 100)),)), home, time[0])
    assert army.intel.enemy_base is None
    army.intel.update(radar, home, time[0])
    assert army.intel.enemy_base == radar.enemy[0][1]


def test_the_factory_saves_for_a_needed_miner_instead_of_buying_another_rhino():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(ore_refinery=3, war_factory=1)
    world.units_built.update(rhino_tank=6)
    available = ('war_miner', 'rhino_tank')
    options, needs = reflex.unit_options('vehicles', available, 1000, time[0])
    assert options == [] and 'too few miners' in needs[0]                    # wait for 1400
    options, needs = reflex.unit_options('vehicles', available, 1500, time[0])
    assert options[0] == 'war_miner'
    army.threat_at = time[0]                                                  # under attack: tanks now
    options, needs = reflex.unit_options('vehicles', available, 1000, time[0])
    assert options == ['rhino_tank']
    # A raid must also exclude a miner when we can afford it, not just stop saving for one.
    options, needs = reflex.unit_options('vehicles', available, 1500, time[0])
    assert options == ['rhino_tank'] and not any('too few miners' in n for n in needs)
    time[0] += reflex.book['units']['miners']['save_unless_threat_within'] + 1
    options, needs = reflex.unit_options('vehicles', available, 1500, time[0])
    assert options[0] == 'war_miner'


# -- team control (vision): Ctrl+1 at each wave, 1 twice to focus the camera on the army ----------

def test_battle_order_fights_enemy_units_near_the_army():
    from addons.chronodivide.army import battle_order
    kind, point = battle_order(cv2.imread(str(FIXTURES / 'enemy-near-base.png')), LAYOUT.tactical, (425, 368))
    assert kind == 'fight' and point == (359, 316)                   # focus fire: the unit nearest the army


def test_battle_order_sieges_the_nearest_building_when_only_buildings_are_near():
    from addons.chronodivide.army import battle_order
    # the enemy base fixture: small house-color parts of buildings, no units (checked by eye)
    kind, point = battle_order(cv2.imread(str(FIXTURES / 'enemy-base-siege.png')), LAYOUT.tactical, (700, 130))
    assert kind == 'siege' and abs(point[0] - 602) <= 15 and abs(point[1] - 109) <= 15
    for name in ('base-reactors.png', 'frame-play.png'):
        assert battle_order(cv2.imread(str(FIXTURES / name)), LAYOUT.tactical, (425, 368)) is None


class TeamGame(CameraGame):
    """Radar clicks move the camera; 1 pressed twice in one transaction centers it on the army."""

    def __init__(self, world, army_view=(950, 110), centers=True):
        super().__init__(world)
        self.army_view, self.centers = army_view, centers

    def act(self, name, lane, steps, commands=1):
        super().act(name, lane, steps, commands)
        if self.centers and sum(1 for s in steps if s.kind == 'key' and s.keys == ('1',)) == 2:
            x, y = self.army_view
            self.world.set('radar', replace(self.world.radar.value, view=(x - 8, y - 14, x + 9, y + 14)))


def test_campaign_waves_save_the_selected_army_as_team_one():
    world, ctx, runner, army, reflex, executor, time = setup()
    executor = runner.executor = CameraGame(world)
    world.set('radar', online_radar())
    target = world.radar.value.enemy[0][1]
    campaign = army.campaign
    runner.start('army:attack', 'army', attack_wave(ctx, target, 'enemy', campaign=campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    order = [step for step in executor.acts[1][2] if step[0] != 'wait']
    assert [step[2] for step in order[:3]] == [('p',), ('p',), ('ctrl', '1')]      # P twice, then Ctrl+1
    assert order[3][0] == 'click' and campaign.team_formed


@pytest.mark.parametrize('stay', [False, True])
def test_a_battle_session_jumps_to_the_front_and_keeps_commanding_the_team(stay):
    from addons.chronodivide.army import battle_session
    world, ctx, runner, army, reflex, executor, time = setup()
    executor = runner.executor = TeamGame(world)
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home), (6, (942, 122)), (3, (930, 100)))))
    ctx.radar_home = home
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-near-base.png'))
    draw_bar(ctx.last_canvas, 407, 340, 36)  # the selected team is visibly present
    campaign = army.campaign
    campaign.target = (945, 125)
    runner.start('army:focus', 'army', battle_session(ctx, campaign, stay=stay))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    acts = [a[2] for a in executor.acts]
    jump, rounds, back = acts[0], acts[1:-1], acts[-1]
    assert jump[0][3] == 'right' and jump[1][:2] == ('click', (942, 122))          # nothing selected, radar click
    assert jump[2][:2] == ('move', ctx.radar_park)                                  # the cursor stays on the radar
    select, order = rounds[0], rounds[1]
    assert select[0][3] == 'right' and select[1][:3] == ('key', None, ('1',))      # deselect, then the field army
    click = next(step for step in order if step[0] == 'click' and step[3] == 'left')
    assert click[2] == ('ctrl', 'shift') and click[1] == (359, 316)          # acquire enemies near the color point
    orders = [a for a in rounds if any(step[2] == ('ctrl', 'shift') for step in a)]
    assert len(orders) >= 5                                                         # it stays and keeps ordering
    if stay:
        assert ctx.camera == 'away'
        assert not any(step[2] == ('h',) for act in acts for step in act)
        executor.acts.clear()
        runner.start('army:focus', 'army', battle_session(ctx, campaign, stay=True))
        assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
        assert any(step[2] == ('ctrl', 'shift') for act in executor.acts for step in act[2])
    else:
        assert back[0][3] == 'right' and back[1][2] == ('h',) and ctx.camera == 'home'
    assert any(e['kind'] == 'order' and e.get('order') == 'focus' and e.get('fight') == 'fight'
               for e in world.events)


def test_without_radar_the_double_tap_focuses_and_repeated_misses_turn_teams_off():
    from addons.chronodivide.army import battle_session
    world, ctx, runner, army, reflex, executor, time = setup()
    executor = runner.executor = TeamGame(world, centers=False)
    world.set('radar', replace(online_radar(), online=False))
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-near-base.png'))
    campaign = army.campaign
    runner.start('army:focus', 'army', battle_session(ctx, campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    taps = [step for step in executor.acts[0][2] if step[0] == 'key']
    assert taps == [('key', None, ('1',), None)] * 2                                # 1, short pause, 1
    runner.start('army:focus', 'army', battle_session(ctx, campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert not campaign.teams and campaign.focus_misses == 2                       # the camera never moved
    assert all(step[2] != ('ctrl', 'shift') for act in executor.acts for step in act[2])   # no blind orders
    world.set('radar', online_radar())
    target = world.radar.value.enemy[0][1]
    executor.acts.clear()
    runner.start('army:attack', 'army', attack_wave(ctx, target, 'enemy', campaign=campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert all(step[2] != ('ctrl', '1') for act in executor.acts for step in act[2])


def test_the_attack_focuses_on_the_army_between_waves():
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    executor = runner.executor = TeamGame(world)
    started = time[0]
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.waves == 1 and army.campaign.team_formed
    assert time[0] - started < 2.5
    sent = army.campaign.last_wave
    time[0] = sent + 2.5
    assert army.step(world.snapshot())[0] == 'army:attack'                          # right after a wave: no focus
    time[0] = sent + 3.5
    assert army.step(world.snapshot())[0] == 'army:focus' and runner.busy('army')
    focused = time[0]
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.waves == 1                                                  # a focus is not a wave
    time[0] = focused + 8.5
    assert army.step(world.snapshot())[0] == 'army:attack'                          # a session every 15 s
    time[0] = focused + 15.5
    assert army.step(world.snapshot())[0] == 'army:focus'



def test_the_factory_saves_for_the_second_refinery_unless_the_base_is_attacked():
    world, ctx, runner, army, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, ore_refinery=1, war_factory=1)
    world.units_built.update(rhino_tank=reflex.book['units']['second_refinery']['after_tanks'], war_miner=1)
    available = ('war_miner', 'rhino_tank')
    options, needs = reflex.unit_options('vehicles', available, 2500, time[0])
    assert options == [] and 'second refinery' in needs[0]                 # 2000 + a Rhino not there yet
    options, needs = reflex.unit_options('vehicles', available, 3000, time[0])
    assert options == ['rhino_tank']                                        # enough for both
    army.threat_at = time[0]
    options, needs = reflex.unit_options('vehicles', available, 1000, time[0])
    assert options == ['rhino_tank']                                        # under attack: tanks now
    army.threat_at = -1e9
    world.structures['ore_refinery'] = 2
    assert not reflex.saving_for_refinery(1000, time[0])                    # built: no more saving



def test_between_reinforcements_waves_command_the_field_army_only():
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    executor = runner.executor = CameraGame(world)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    first = [step[2] for act in executor.acts for step in act[2] if step[0] in ('key', 'chord')]
    assert first[:3] == [('p',), ('p',), ('ctrl', '1')]                        # all of it, saved as team 1
    executor.acts.clear()
    time[0] += 41                                                               # the timer wave
    army.campaign.last_focus = time[0]  # the field army has already been inspected
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    keys = [step[2] for act in executor.acts for step in act[2] if step[0] in ('key', 'chord')]
    assert ('1',) in keys and ('p',) not in keys                               # team 1 only
    executor.acts.clear()
    world.units_built['rhino_tank'] += 4                                        # enough new units at home
    time[0] += 9                                                                # (waves are 8 s apart)
    army.campaign.last_focus = time[0]
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    keys = [step[2] for act in executor.acts for step in act[2] if step[0] in ('key', 'chord')]
    assert keys[:3] == [('p',), ('p',), ('ctrl', '1')]                          # they join the team


@pytest.mark.parametrize('queued,ready,merge', [(4, 0, False), (0, 4, True)])
def test_reinforcements_follow_ready_home_units_instead_of_queued_totals(queued, ready, merge):
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    executor = runner.executor = CameraGame(world)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    previous_merge = army.campaign.units_at_merge
    world.units_built['rhino_tank'] += queued
    time[0] += 41
    army.campaign.last_focus = time[0]
    army.census_enabled = True
    army.census_tanks, army.census_at = ready, time[0]
    executor.acts.clear()

    assert army._attack(time[0], True)[0] == 'army:attack'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    keys = [step[2] for act in executor.acts for step in act[2] if step[0] in ('key', 'chord')]
    assert (('ctrl', '1') in keys) is merge
    assert (('p',) in keys) is merge
    if not merge:
        assert army.campaign.units_at_merge == previous_merge
    else:
        assert army.census_tanks is None  # consume the observation after these units are sent


def test_active_attack_refreshes_the_home_census_before_merging_reinforcements():
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    runner.executor = CameraGame(world)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    time[0] += 41
    army.campaign.last_focus = time[0]
    army.census_enabled = True
    army.census_tanks, army.census_at = 4, time[0] - army.book['army']['census']['during_attack'] - 1
    assert army._attack(time[0], True) == ('army:census', None)
    assert runner.tasks['army'].name == 'army:census'


def test_delayed_controller_observes_sent_army_before_another_home_census_or_wave():
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    runner.executor = TeamGame(world)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    time[0] += 41  # slower input has missed the normal between-wave window
    army.census_enabled = True
    army.census_tanks, army.census_at = 4, time[0] - army.book['army']['census']['during_attack'] - 1
    assert army._attack(time[0], True)[0] == 'army:focus'
    assert runner.tasks['army'].name == 'army:focus'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.campaign.waves == 1
    # After observing the fight, home readiness is refreshed before reinforcements.
    assert army._attack(time[0], True) == ('army:census', None)


def test_failed_prism_reinforcement_staging_observes_the_existing_field_army():
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    runner.executor = TeamGame(world)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    time[0] += 41
    army.campaign.last_focus = time[0] - 20
    army.campaign.prism_staging_failed = True
    army.census_enabled = True
    army.census_tanks, army.census_at = 8, time[0]
    assert army._attack(time[0], True)[0] == 'army:focus'
    assert runner.tasks['army'].name == 'army:focus'
    assert not army.campaign.prism_staging_failed


@pytest.mark.parametrize('sieging', [False, True])
def test_healthy_allied_siege_keeps_its_typed_groups_instead_of_merging_home_reinforcements(sieging):
    from addons.chronodivide import playbook
    world, ctx, runner, army, reflex, executor, time = attack_setup()
    army.book = playbook.load('addons/chronodivide/profiles/startlux-ra2-allied.json')
    world.knowledge = playbook.knowledge(army.book)
    world.units_built['prism_tank'] = 4
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.target = world.radar.value.enemy[0][1]
    campaign.kind = 'enemy'
    campaign.started = time[0] - 60
    campaign.waves = 1
    campaign.last_wave = time[0] - 10
    campaign.last_focus = time[0] - 8
    campaign.army_seen_at = time[0]
    campaign.sieging = sieging
    campaign.prism_roles = {'prism_tank': '5', 'mirage_tank': '6'}
    army.census_enabled = True
    army.census_tanks, army.census_at = 8, time[0]
    army._attack(time[0], True)
    assert runner.tasks['army'].name == ('army:focus' if sieging else 'army:attack')
    assert campaign.prism_roles == {'prism_tank': '5', 'mirage_tank': '6'}



def test_a_battle_session_goes_home_early_for_a_building_to_place():
    from addons.chronodivide.army import battle_session
    world, ctx, runner, army, reflex, executor, time = setup()
    executor = runner.executor = TeamGame(world)
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home), (6, (942, 122)))))
    ctx.radar_home = home
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-near-base.png'))
    draw_bar(ctx.last_canvas, 407, 340, 36)  # the selected team is visibly present
    world.structures['construction_yard'] = 1
    world.update_sidebar('buildings', [Slot(0, 'tesla_reactor', 'ready', 1.0)])
    runner.start('army:focus', 'army', battle_session(ctx, army.campaign, leave=army._leave_battle))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    orders = [a for a in executor.acts if any(step[2] == ('ctrl', 'shift') for step in a[2])]
    assert len(orders) == 3                                  # a few orders, then home for the placement


def test_after_the_enemy_base_is_razed_the_remaining_buildings_are_hunted():
    radar = online_radar()
    home = radar.own[0][1]
    base = (971, 108)
    razed = replace(radar, enemy=((3, (905, 140)),), explored=np.ones_like(radar.explored))
    assert choose_target(razed, home, known=base) == ((905, 140), 'enemy')      # a lone building elsewhere
    before = replace(radar, enemy=((3, (905, 140)),), explored=np.zeros_like(radar.explored))
    assert choose_target(before, home)[1] == 'guess'                                # base unknown: no chasing


def test_battle_drops_a_target_that_disappears_while_selecting_the_team():
    from addons.chronodivide.army import battle_session
    world, ctx, runner, army, reflex, executor, time = setup()

    class DepartingEnemy(TeamGame):
        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            if any(step.kind == 'key' and step.keys == ('1',) for step in steps):
                ctx.last_canvas = cv2.imread(str(FIXTURES / 'base-reactors.png'))

    executor = runner.executor = DepartingEnemy(world)
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home), (6, (942, 122)))))
    ctx.radar_home = home
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-near-base.png'))
    runner.start('army:focus', 'army', battle_session(ctx, army.campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    assert not any(step[0] == 'click' and step[2] for act in executor.acts for step in act[2])
    note = next(e for e in world.events if e.get('order') == 'focus')
    assert note['orders'] == []


def test_a_controller_joining_a_running_match_assumes_the_army_and_a_base():
    from addons.chronodivide.player import attach_to_match
    world = World()
    attach_to_match(world, '40')
    assert world.units_built['rhino_tank'] == 40 and world.structures['war_factory'] == 1
    assert world.mcv_deployed.value
    fresh = World()
    attach_to_match(fresh, None)
    assert not fresh.units_built and not fresh.structures


def test_no_battle_session_while_the_army_is_still_at_home():
    from addons.chronodivide.army import battle_session
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    runner.start('army:focus', 'army', battle_session(ctx, army.campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert executor.acts == [] and army.campaign.teams and army.campaign.focus_misses == 0



from addons.chronodivide.tests.test_unitboxes import draw_bar  # noqa: E402


class CensusGame(CameraGame):
    """P selects the units at home: the next canvas shows their health bars ((frame length, health))."""

    def __init__(self, world, ctx, bars):
        super().__init__(world)
        self.ctx, self.bars = ctx, bars

    def act(self, name, lane, steps, commands=1):
        super().act(name, lane, steps, commands)
        if any(s.kind == 'key' and s.keys == ('p',) for s in steps):
            canvas = np.full((768, 1024, 3), (40, 120, 90), np.uint8)          # grass
            for i, (length, health) in enumerate(self.bars):
                draw_bar(canvas, 100 + 60 * (i // 10), 200 + 40 * (i % 10), length, health)
            self.ctx.last_canvas = canvas


def test_the_attack_waits_for_a_count_of_the_army_at_home():
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=14)
    army.census_enabled = True
    # Five tanks (two badly hurt: their lit pips are shorter than an infantry bar) and six infantry.
    runner.executor = CensusGame(world, ctx, [(36, 1.0)] * 3 + [(36, 0.2)] * 2 + [(18, 0.9)] * 6)
    assert army.step(world.snapshot())[0] == 'army:census'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.census_tanks == 5.6 and army.available() == 5
    time[0] += 1.5
    assert army.step(world.snapshot()) is None and not army.campaign.active   # 14 built, 5 left: no attack
    assert army.intent.posture == 'gather'
    assert any(e['kind'] == 'census' and e['vehicles'] == 5 for e in world.events)


@pytest.mark.parametrize('phase', ['regrouped', 'campaign', 'unregrouped'])
@pytest.mark.parametrize('home,group', [(12, 17), (17, 12), (12, 12)])
def test_census_centers_regrouped_army_but_keeps_attack_reinforcements_separate(phase, home, group):
    from addons.chronodivide.army import census

    world, ctx, runner, army, reflex, executor, time = attack_setup(units=20)
    army.regrouped_defense = None if phase == 'unregrouped' else time[0]
    army.campaign.active = phase == 'campaign'

    class EdgeCensusGame(CensusGame):
        def __init__(self):
            super().__init__(world, ctx, [])
            self.focused, self.group_keys = False, 0

        def act(self, name, lane, steps, commands=1):
            for step in steps:
                if step.kind == 'key' and step.keys == ('h',):
                    self.focused = False
                elif step.kind == 'key' and step.keys == ('1',):
                    self.group_keys += 1
                    self.focused = self.group_keys >= 2
            self.bars = [(36, 0.9)] * (group if self.focused else home)
            super().act(name, lane, steps, commands)

    game = runner.executor = EdgeCensusGame()
    runner.start('army:census', 'army', census(ctx, army))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert army.census_tanks == (max(home, group) if phase == 'regrouped' else home)
    assert game.group_keys == (2 if phase == 'regrouped' else 0)
    assert ctx.camera == 'home' and not game.focused


def test_a_counted_army_that_is_big_enough_attacks():
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=14)
    army.census_enabled = True
    runner.executor = CensusGame(world, ctx, [(36, 0.8)] * 13)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    time[0] += 1.5
    assert army.step(world.snapshot())[0] == 'army:attack' and army.campaign.active


class ScoutGame(Recorder):
    """P selects the two dogs at home: the next canvas shows two infantry health bars."""

    def __init__(self, world, ctx):
        super().__init__()
        self.ctx = ctx

    def act(self, name, lane, steps, commands=1):
        super().act(name, lane, steps, commands)
        if any(s.kind == 'key' and s.keys == ('p',) for s in steps):
            canvas = np.full((768, 1024, 3), (40, 120, 90), np.uint8)
            draw_bar(canvas, 400, 300, 18)
            draw_bar(canvas, 440, 310, 18)
            self.ctx.last_canvas = canvas


def test_dog_scouts_are_grouped_as_team_two_while_other_infantry_waits_then_explore_the_base():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    executor = runner.executor = ScoutGame(world, ctx)
    ctx.layout.points['waypoint'] = (362, 754)
    world.set('credits', 3000)
    world.units_built['attack_dog'] = 2
    assert army.step(world.snapshot())[0] == 'army:scouts'
    assert reflex.next_item('infantry', ('conscript', 'attack_dog'), time[0]) is None     # conscripts wait
    assert run(runner, world, time, lambda: not runner.busy('army'))
    keys = [step[2] for act in executor.acts for step in act[2] if step[0] in ('key', 'chord')]
    assert ('ctrl', '2') in keys and army.scouts_ready
    assert reflex.next_item('infantry', ('conscript', 'attack_dog'), time[0]) == 'conscript'
    assert 'explore' in reflex.feasible_intel(time[0])                                  # no radar needed
    army.intent.set('intel', 'explore', 'systemone')
    executor.acts.clear()
    time[0] += 2
    assert army.step(world.snapshot())[0] == 'army:explore'
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=400)
    moves = [a[2] for a in executor.acts if any(step[2] == ('2',) for step in a[2])]
    assert len(moves) == 2 and army.exploring and not army.explored
    assert sum(step[0] == 'route' for a in executor.acts for step in a[2]) == 1
    assert 'explore' not in reflex.feasible_intel(time[0])


def test_finite_exploration_releases_the_lane_before_construction_finishes():
    from addons.chronodivide.army import explore_route
    from addons.chronodivide.executor import Step

    for lane, item in (('buildings', 'war_factory'), ('defenses', 'sentry_gun')):
        world, ctx, runner, army, reflex, executor, time = intel_setup()
        ctx.layout.points['waypoint'] = (362, 754)
        executor = runner.executor = ScoutGame(world, ctx)
        executor.act('select', 'army', [Step.key('p')])
        runner.start('army:explore', 'army', explore_route(ctx, army, '2', [(200, 0), (0, 200)], 4))
        assert run(runner, world, time, lambda: not runner.busy('army'), ticks=30)
        assert army.exploring and not army.explored
        world.update_sidebar(lane, [Slot(0, item, 'ready', 1.0)])
        assert reflex.plan(world.snapshot(), time[0])[0] == f'place:{item}'


def test_dog_scout_request_waits_for_existing_exploration_return():
    world, ctx, runner, army, reflex, executor, time = intel_setup()
    radar = online_radar()
    world.set('radar', replace(radar, enemy=(), explored=np.zeros_like(radar.explored)))
    ctx.radar_home = radar.own[0][1]
    army.scouts_ready = army.exploring = True
    army.intent.set('intel', 'scout', 'systemone')
    assert army._intel(time[0], True) is None
    assert army.intent.intel == 'scout' and army.exploring and not army.explore_failed
    assert not runner.tasks
    army.exploring, army.explored = False, True
    assert army._intel(time[0], True)[0] == 'army:scout'
    assert runner.tasks['army'].name == 'army:scout' and army.intent.intel is None
