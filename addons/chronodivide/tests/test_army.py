"""Army control and placement-ghost perception on frames recorded in a live game (Chrome, 2026-09-24)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.army import Army, threat
from addons.chronodivide.executor import Trust
from addons.chronodivide.layout import Layout
from addons.chronodivide.perception.placement import read_ghost
from addons.chronodivide.skills import Context
from addons.chronodivide.tasks import TaskRunner
from addons.chronodivide.world import World

FIXTURES = Path(__file__).with_name('fixtures')


def frame(name):
    return cv2.imread(str(FIXTURES / name))


def test_no_threat_in_our_own_base_with_glowing_tesla_reactors():
    assert threat(frame('base-reactors.png')) is None
    assert threat(frame('frame-play.png')) is None


def test_native_lone_raider_is_confirmed_even_when_captures_arrive_faster_than_half_a_second():
    from addons.chronodivide.client import ClientProfile
    from addons.chronodivide.perception import colors
    previous = ClientProfile.load('chronodivide-allied').use_colors()
    try:
        world, ctx, runner, army, executor, time = setup()
        ctx.compressed_frames = True
        ctx.last_canvas = frame('allied-lone-raider-native.png')
        expected = threat(ctx.last_canvas, min_blob=12, min_total=24, compressed=True)
        assert expected is not None
        assert threat(ctx.last_canvas, compressed=True) is None
        for elapsed in (0, 0.1, 0.2, 0.3, 0.4, 0.6):
            army._look(time[0] + elapsed)
        assert army.seen == expected
    finally:
        colors.use(*previous)


def test_compressed_standing_gi_group_is_a_fresh_attack_move_target():
    from addons.chronodivide.army import raider_target
    from addons.chronodivide.perception.units import enemy_blobs
    canvas = frame('standing-gi-raid-browser.png')
    tactical = Layout.for_canvas(1024, 768).tactical
    assert enemy_blobs(canvas, min_area=8, close=3, compressed=True) == []
    assert raider_target(canvas, tactical, (320, 513), 600, compressed=True) == (502, 463)


def test_deployed_gi_seen_by_the_observer_is_also_a_home_defense_target():
    from addons.chronodivide.army import raider_target
    from addons.chronodivide.perception.enemy_threats import deployed_gi_targets
    from addons.chronodivide.perception.units import enemy_blobs
    canvas = frame('deployed-gi-home-raid-browser.png')
    tactical = Layout.for_canvas(1024, 768).tactical
    assert enemy_blobs(canvas, min_area=8, close=3, compressed=True) == []
    assert deployed_gi_targets(canvas, (0, 0, 850, 736), compressed=True) == [(772, 392)]
    assert raider_target(canvas, tactical, (425, 368), 600, compressed=True) == (772, 392)
    assert raider_target(canvas, tactical, (100, 100), 300, compressed=True) is None


def test_standing_gi_fragment_group_starts_defense_only_after_fresh_confirmation():
    world, ctx, runner, army, executor, time = setup()
    world.units_built['rhino_tank'] = 3
    ctx.compressed_frames = True
    ctx.last_canvas = frame('standing-gi-raid-browser.png')
    assert threat(ctx.last_canvas, min_blob=12, min_total=24, compressed=True) is None
    assert army.step(world.snapshot()) is None
    assert army.seen is None
    time[0] += 1
    assert army.step(world.snapshot())[0] == 'army:defend'
    assert army.seen[0] == (553, 459) and army.seen[1] >= 12
    # A remembered cluster cannot create another current detection after it disappears.
    ctx.last_canvas = np.zeros_like(ctx.last_canvas)
    army._look(time[0] + 1)
    assert army.seen is None


@pytest.mark.parametrize('name', ['base-reactors.png', 'reactor-glow-browser.png', 'frame-play.png'])
def test_repeated_own_base_pixels_do_not_confirm_fragment_threat(name):
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    ctx.last_canvas = frame(name)
    for elapsed in (0, 1, 2):
        army._look(time[0] + elapsed)
        assert army.seen is None and army._small_threat is None


def test_defense_does_not_drop_visible_standing_gis_after_selecting_defenders():
    from addons.chronodivide.army import defense_session
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    ctx.last_canvas = frame('standing-gi-raid-browser.png')
    runner.start('army:defend', 'army', defense_session(ctx, 1, 1, 600))
    for _ in range(20):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.tasks:
            break
    note = next(e for e in world.events if e.get('order') == 'defense')
    assert note['defenders'] == 6 and note['orders'] == 1
    orders = [s for _, _, steps in executor.acts for s in steps if s[0] == 'click' and s[2]]
    assert orders == [('click', (502, 463), ('ctrl', 'shift'))]


def standing_raid(shift=0):
    # Independent screenshot after selecting the damaged refinery, canvas crop (680,425,95,95).
    canvas = np.zeros((768, 1024, 3), np.uint8)
    canvas[425:520, 680 + shift:775 + shift] = frame('standing-gi-selected-heldout.png')
    return canvas


def test_appearance_confirms_live_standing_raid_with_fresh_point_not_missing_color():
    from addons.chronodivide.army import _compressed_raider_groups, battle_order, raider_target, siege_target
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    world.units_built['rhino_tank'] = 3
    ctx.last_canvas = standing_raid()
    tactical = ctx.layout.tactical
    assert threat(ctx.last_canvas, min_blob=12, min_total=24, compressed=True) is None
    assert _compressed_raider_groups(ctx.last_canvas, tactical, tactical.center, 1000) == []
    assert army.step(world.snapshot()) is None
    assert army.seen is None
    # A blank sample retains history but cannot establish a threat.
    ctx.last_canvas = np.zeros_like(ctx.last_canvas)
    time[0] += 1
    assert army.step(world.snapshot()) is None and army.seen is None
    ctx.last_canvas = standing_raid(10)
    time[0] += 1
    assert army.step(world.snapshot())[0] == 'army:defend'
    assert army.seen == ((718, 462), 0)  # no fabricated house-color pixel count
    assert raider_target(ctx.last_canvas, tactical, tactical.center, 600, compressed=True) == (718, 462)
    assert battle_order(ctx.last_canvas, tactical, (600, 450), compressed=True) == ('fight', (718, 462))
    assert siege_target(ctx.last_canvas, tactical, (600, 450), compressed=True) is None
    assert raider_target(np.zeros_like(ctx.last_canvas), tactical, tactical.center, 600, compressed=True) is None


@pytest.mark.parametrize('interruption', ['away', 'menu', 'native'])
def test_standing_appearance_cannot_confirm_outside_compressed_home_hud(interruption):
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = interruption != 'native'
    ctx.last_canvas = standing_raid()
    army._look(time[0])
    if interruption == 'away':
        ctx.camera = 'away'
    elif interruption == 'menu':
        world.set('mode', 'menu')
    army._look(time[0] + 1)
    assert army.seen is None and army._small_threat is None


def test_compressed_group_target_rejects_isolated_specks_and_sparse_fragments():
    from addons.chronodivide.army import raider_target
    tactical = Layout.for_canvas(1024, 768).tactical
    canvas = np.zeros((768, 1024, 3), np.uint8)
    for x in range(400, 480, 10):
        canvas[400, x] = (212, 105, 34)
    assert raider_target(canvas, tactical, (425, 368), 600, compressed=True) is None
    for x in (410, 440, 470):
        canvas[430, x:x + 3] = (212, 105, 34)
    assert raider_target(canvas, tactical, (425, 368), 600, compressed=True) is None
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames, ctx.last_canvas = True, canvas
    for elapsed in (0, 1, 2):
        army._look(time[0] + elapsed)
        assert army.seen is None and army._small_threat is None


def test_enemy_tanks_near_the_base_are_a_threat_with_a_target_among_them():
    point, pixels = threat(frame('enemy-near-base.png'))
    assert pixels >= 60
    assert 0 <= point[0] < 850 and 0 <= point[1] < 736


@pytest.mark.parametrize('quality', [70, 80, 90, 95])
def test_compressed_enemy_targets_survive_without_confusing_reactor_glow(quality):
    from addons.chronodivide.army import battle_order, raider_target, siege_target
    from addons.chronodivide.intel import survey
    tactical = Layout.for_canvas(1024, 768).tactical

    def jpeg(name):
        data = cv2.imencode('.jpg', frame(name), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    raid = jpeg('enemy-near-base.png')
    assert threat(raid, compressed=True) is not None
    assert raider_target(raid, tactical, (425, 368), 600, compressed=True) is not None
    kind, point = battle_order(raid, tactical, (425, 368), compressed=True)
    assert kind == 'fight' and abs(point[0] - 359) <= 4 and abs(point[1] - 316) <= 4
    assert survey(raid, tactical, compressed=True)['enemy_pixels'] >= 50
    base = jpeg('enemy-base-siege.png')
    point = siege_target(base, tactical, (740, 120), compressed=True)
    assert point is not None and abs(point[0] - 602) <= 15 and abs(point[1] - 109) <= 15
    for name in ('base-reactors.png', 'frame-play.png', 'frame-start.png'):
        quiet = jpeg(name)
        assert threat(quiet, compressed=True) is None
        assert raider_target(quiet, tactical, (425, 368), 600, compressed=True) is None
        assert survey(quiet, tactical, compressed=True)['enemy_objects'] == 0


def test_army_uses_compressed_context_for_raid_detection():
    world, ctx, runner, army, executor, time = setup()
    encoded = cv2.imencode('.jpg', frame('enemy-near-base.png'), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
    ctx.last_canvas = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert threat(ctx.last_canvas) is None               # strict native ratio remains unchanged
    ctx.compressed_frames = True
    army.step(world.snapshot())
    assert army.seen is not None


def test_live_compressed_reactor_edges_are_not_raider_targets():
    from addons.chronodivide.perception.units import enemy_blobs
    from addons.chronodivide.army import raider_target
    from addons.chronodivide.layout import Box
    # Live browser frame 1790576565750122300, cropped at (150,350). The old hue mask
    # produced an 8-pixel enemy at (200,403) on our reactor, closer than the real raid.
    image = frame('reactor-glow-browser.png')
    height, width = image.shape[:2]
    region = (0, 0, width, height)
    assert enemy_blobs(image, region, min_area=8, close=3, compressed=True) == []
    assert raider_target(image, Box(*region), (width // 2, height // 2), 600, compressed=True) is None


def test_compressed_reactor_edge_does_not_draw_defenders_away_from_real_tank():
    from addons.chronodivide.army import raider_target
    from addons.chronodivide.layout import Box
    from addons.chronodivide.perception.units import enemy_blobs
    # 2026-09-30 live frame: crop (170,350)-(680,550). The reactor edge at
    # (30,43) pulled the army west while an enemy tank attacked from the east.
    image = frame('reactor-edge-and-raider-browser.png')
    region = (0, 0, image.shape[1], image.shape[0])
    blobs = enemy_blobs(image, region, min_area=8, close=3, compressed=True)
    assert blobs and all(point[0] > 400 for _, point, _ in blobs)
    target = raider_target(image, Box(*region), (120, 110), 600, margin=0, compressed=True)
    assert target is not None and target[0] > 400


def test_small_jpeg_raid_needs_two_fresh_samples_then_defends_before_census():
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    world.units_built['rhino_tank'] = 3
    ctx.last_canvas = frame('small-raid-browser-a.png')
    assert threat(ctx.last_canvas, compressed=True) is None
    army.step(world.snapshot())
    assert army.seen is None and not runner.busy('army')
    time[0] += 1.0
    ctx.last_canvas = frame('small-raid-browser-b.png')
    current = threat(ctx.last_canvas, min_blob=12, min_total=24, compressed=True)
    army.census_enabled = True
    assert army.step(world.snapshot())[0] == 'army:defend'
    assert army.seen == current and army.threat_at == time[0]
    assert runner.tasks['army'].name == 'army:defend'
    assert not army._census_due(time[0])
    world.units_built['attack_dog'] = 2
    assert not army._scouts_due(time[0])


@pytest.mark.parametrize('missing', ['base-reactors.png', 'reactor-glow-browser.png'])
def test_small_jpeg_contact_confirms_after_brief_missing_frame_only_with_fresh_evidence(missing):
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    world.units_built['rhino_tank'] = 3
    # Live failed raid: a contact at +458.88s reappeared at +461.31s, with a miss between.
    time[0] = 458.88
    ctx.last_canvas = frame('small-raid-browser-a.png')
    army.step(world.snapshot())
    remembered = army._small_threat
    time[0] = 460.23
    ctx.last_canvas = frame(missing)
    army.step(world.snapshot())
    assert army.seen is None and army.threat_at == -1e9
    assert army._small_threat == remembered and not runner.busy('army')
    time[0] = 461.31
    ctx.last_canvas = frame('small-raid-browser-b.png')
    current = threat(ctx.last_canvas, min_blob=12, min_total=24, compressed=True)
    assert army.step(world.snapshot())[0] == 'army:defend'
    assert army.seen == current and army.seen[0] != remembered[0]
    assert army.threat_at == time[0]


def test_missing_jpeg_contact_expires_without_extending_confirmation_window():
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    ctx.last_canvas = frame('small-raid-browser-a.png')
    army._look(time[0])
    ctx.last_canvas = frame('base-reactors.png')
    for elapsed in (1.0, 2.0, 3.01):
        army._look(time[0] + elapsed)
        assert army.seen is None and army.threat_at == -1e9
    assert army._small_threat is None
    ctx.last_canvas = frame('small-raid-browser-b.png')
    army._look(time[0] + 3.1)
    assert army.seen is None


def test_small_jpeg_confirmation_still_requires_half_second_separation():
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    ctx.last_canvas = frame('small-raid-browser-a.png')
    army._look(time[0])
    ctx.last_canvas = frame('base-reactors.png')
    army._look(time[0] + 0.1)
    ctx.last_canvas = frame('small-raid-browser-b.png')
    army._look(time[0] + 0.2)
    assert army.seen is None and army.threat_at == -1e9


@pytest.mark.parametrize('interruption', ['away', 'menu', 'distant', 'gap'])
def test_small_jpeg_confirmation_does_not_survive_unrelated_or_expired_evidence(interruption):
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    first = frame('small-raid-browser-a.png')
    ctx.last_canvas = first
    army._look(time[0])
    assert army.seen is None
    time[0] += 1.0
    if interruption == 'away':
        ctx.camera = 'away'
    elif interruption == 'menu':
        world.set('mode', 'menu')
    elif interruption == 'distant':
        ctx.last_canvas = np.pad(first, ((0, 0), (200, 0), (0, 0)))
    else:
        time[0] += 3.0
    army._look(time[0])
    assert army.seen is None and army.threat_at == -1e9
    ctx.camera = 'home'
    world.set('mode', 'hud')
    ctx.last_canvas = frame('small-raid-browser-b.png')
    time[0] += 1.0
    army._look(time[0])
    if interruption != 'gap':
        assert army.seen is None


def test_repeated_native_small_contact_stays_below_existing_threshold():
    world, ctx, runner, army, executor, time = setup()
    ctx.last_canvas = frame('small-raid-browser-a.png')
    army._look(time[0])
    time[0] += 1.0
    ctx.last_canvas = frame('small-raid-browser-b.png')
    army._look(time[0])
    assert army.seen is None and army._small_threat is None


def test_flickering_raid_gets_next_free_army_slot_without_firing_at_stale_contact():
    world, ctx, runner, army, executor, time = setup()
    world.units_built.update(rhino_tank=3, attack_dog=2)
    army.census_enabled = True
    army.exploring, army.explore_deadline = True, time[0] + 60
    ctx.last_canvas = frame('enemy-near-base.png')
    assert army.step(world.snapshot(), act=False)[0] == 'army:defend'
    assert not army.exploring and army.explore_failed and not army.explored
    time[0] += 1.1
    ctx.last_canvas = frame('base-reactors.png')
    assert army.step(world.snapshot()) == ('army:defend', None)
    assert not army._census_due(time[0]) and not army._scouts_due(time[0])
    for _ in range(20):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.busy('army'):
            break
    note = next(e for e in world.events if e.get('order') == 'defense')
    assert note['orders'] == 0
    assert not any(step[0] == 'click' and step[2] for _, _, act in executor.acts for step in act)
    time[0] = army.threat_at + 3.1
    assert not army._recent_threat(time[0])
    assert army._census_due(time[0]) and army._scouts_due(time[0])


def test_war_factory_footprint_blocked_on_its_lower_right():
    ghost = read_ghost(frame('ghost-war-factory.png'), frame('ghost-reference.png'), (262, 258))
    assert ghost.seen and not ghost.valid
    assert ghost.green > 5000 and ghost.red > 500
    assert ghost.red_center[0] > ghost.green_center[0] and ghost.red_center[1] > ghost.green_center[1]
    none = read_ghost(frame('ghost-reference.png'), frame('ghost-reference.png'), (262, 258))
    assert not none.seen


class Recorder:
    def __init__(self):
        self.acts = []

    def act(self, name, lane, steps, commands=1):
        self.acts.append((name, lane, [(s.kind, s.point, s.keys) for s in steps]))


def setup():
    time = [100.0]
    world = World(clock=lambda: time[0])
    world.set('mode', 'hud')
    ctx = Context(Layout.for_canvas(1024, 768), world)
    executor = Recorder()
    runner = TaskRunner(executor, Trust(), clock=lambda: time[0])
    army = Army(ctx, runner, clock=lambda: time[0])
    army.census_enabled = False          # counted in its own tests
    return world, ctx, runner, army, executor, time


class DefenseGame(Recorder):
    """Raiders beside our base; P selects our three tanks there, which then show health bars."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx, self.base = ctx, frame('enemy-near-base.png')
        ctx.last_canvas = self.base.copy()

    def act(self, name, lane, steps, commands=1):
        super().act(name, lane, steps, commands)
        canvas = self.base.copy()
        if any(s.kind == 'key' and s.keys == ('p',) for s in steps):
            from addons.chronodivide.tests.test_unitboxes import draw_bar
            for x, y in ((300, 400), (345, 425), (390, 400)):
                draw_bar(canvas, x, y, 36, 0.9)
        self.ctx.last_canvas = canvas


def test_raiders_in_view_get_attack_move_without_forcing_fire_on_color_points():
    world, ctx, runner, army, executor, time = setup()
    executor = runner.executor = DefenseGame(ctx)
    army.step(world.snapshot())
    assert not runner.busy('army') and any(e['kind'] == 'attack' for e in world.events)   # no army yet
    world.units_built['rhino_tank'] = 3
    time[0] += 1.5
    assert army.step(world.snapshot())[0] == 'army:defend'
    for _ in range(90):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.busy('army'):
            break
    steps = [step for _, _, act in executor.acts for step in act]
    assert steps[1] == ('key', None, ('h',))                                        # camera home first
    fire = [step for step in steps if step[0] == 'click' and step[2] == ('ctrl', 'shift')]
    assert len(fire) >= 2                                                            # round after round
    assert not any(step[0] == 'click' and step[2] == ('ctrl',) for step in steps)
    note = next(e for e in world.events if e.get('order') == 'defense')
    assert note['defenders'] == 3 and note['orders'] == len(fire)
    time[0] += 1.0
    assert army.step(world.snapshot())[0] == 'army:defend' and runner.busy('army')   # raiders remain: again


def test_home_defense_commands_selected_tanks_toward_the_recorded_deployed_gi():
    from addons.chronodivide.army import defense_session
    world, ctx, runner, army, executor, time = setup()
    ctx.compressed_frames = True
    executor = runner.executor = DefenseGame(ctx)
    executor.base = frame('deployed-gi-home-raid-browser.png')
    ctx.last_canvas = executor.base.copy()
    runner.start('army:defend', 'army', defense_session(ctx, rounds=1, every=0.2, reach=600))
    for _ in range(30):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.busy('army'):
            break
    fire = [step for _, _, act in executor.acts for step in act
            if step[0] == 'click' and step[2] == ('ctrl', 'shift')]
    assert fire == [('click', (772, 392), ('ctrl', 'shift'))]
    note = next(e for e in world.events if e.get('order') == 'defense')
    assert note['orders'] == 1 and note['defenders'] == 3


def test_home_defense_recalls_map_military_before_it_enters_the_home_view():
    from addons.chronodivide.army import defense_session
    from addons.chronodivide.perception.unitboxes import our_units

    world, ctx, runner, army, executor, time = setup()
    ctx.last_canvas = frame('enemy-near-base.png')
    # Remote defenders produce no selected health bars in the home screenshot.
    assert not our_units(ctx.last_canvas)
    runner.start('army:defend', 'army', defense_session(ctx, rounds=1, every=0.2, reach=600))
    for _ in range(30):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.busy('army'):
            break
    steps = [step for _, _, act in executor.acts for step in act]
    assert len([step for step in steps if step == ('key', None, ('p',))]) == 4
    assert len([step for step in steps if step[0] == 'click' and step[2] == ('ctrl', 'shift')]) == 1
    note = next(e for e in world.events if e.get('order') == 'defense')
    assert note['orders'] == 1 and note['defenders'] == 0


@pytest.mark.parametrize('profile', ['startlux-ra2-allied.json', 'startlux-ra2-soviet.json'])
@pytest.mark.parametrize('home', [False, True])
@pytest.mark.parametrize('local_defender', [False, True])
def test_remote_defenders_receive_orders_despite_another_lane_clearing_selection(profile, home, local_defender):
    from addons.chronodivide import playbook
    from addons.chronodivide.army import defense_session
    from addons.chronodivide.executor import Step
    from addons.chronodivide.tasks import Act
    from addons.chronodivide.tests.test_unitboxes import draw_bar

    world, ctx, runner, army, executor, time = setup()
    army.book = playbook.load(Path(__file__).parents[1] / 'profiles' / profile)
    owned = {'distant_tank', 'distant_anti_air'} | ({'local_guard'} if local_defender else set())

    class SelectionGame:
        def __init__(self):
            self.selected, self.orders, self.map_next = set(), [], False

        def act(self, name, lane, steps, commands=1):
            for step in steps:
                if step.kind == 'click' and step.button == 'right':
                    self.selected, self.map_next = set(), False
                elif step.kind == 'key' and step.keys == ('p',):
                    self.selected = owned.copy() if self.map_next or not local_defender else {'local_guard'}
                    self.map_next = True
                elif step.kind == 'key' and step.keys == ('1',):
                    self.selected = {'distant_tank', 'distant_anti_air'}
                    self.map_next = False
                elif step.kind == 'click' and step.keys == ('ctrl', 'shift'):
                    self.orders.append(self.selected.copy())
            ctx.last_canvas = frame('enemy-near-base.png')
            if 'local_guard' in self.selected:
                draw_bar(ctx.last_canvas, 300, 400, 36, 0.9)

    game = runner.executor = SelectionGame()
    ctx.last_canvas = frame('enemy-near-base.png')

    def sidebar():
        while True:
            yield Act([Step.click(ctx.clear_point, 'right')], commands=0)

    runner.start('army:defend', 'army', defense_session(
        ctx, 2, 0.2, 600, home=home, keep_home_infantry=not home, team_formed=not home), priority=2)
    runner.start('sidebar:inspect', 'sidebar', sidebar(), priority=5)
    for _ in range(60):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.busy('army'):
            break
    assert not runner.busy('army')
    expected = owned if home else {'distant_tank', 'distant_anti_air'}
    assert game.orders == [expected, expected]


@pytest.mark.parametrize('home', [False, True])
def test_defense_rechecks_the_threat_view_instead_of_distant_selected_defenders(home):
    from addons.chronodivide.army import defense_session
    from addons.chronodivide.tests.test_unitboxes import draw_bar

    world, ctx, runner, army, executor, time = setup()
    ctx.last_canvas = frame('enemy-near-base.png')
    # A selected vehicle on the opposite edge must not hide the confirmed raid.
    draw_bar(ctx.last_canvas, 800, 700, 36, 0.9)
    runner.start('army:defend', 'army', defense_session(ctx, 1, 0.2, 600, home=home))
    for _ in range(30):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.busy('army'):
            break
    fire = [step for _, _, act in executor.acts for step in act
            if step[0] == 'click' and step[2] == ('ctrl', 'shift')]
    assert fire == [('click', (457, 321), ('ctrl', 'shift'))]
    assert next(e for e in world.events if e.get('order') == 'defense')['defenders'] == 1


def test_defense_does_not_order_toward_an_enemy_that_disappeared_during_selection():
    from addons.chronodivide.army import defense_session
    from addons.chronodivide.tests.test_unitboxes import draw_bar

    world, ctx, runner, army, executor, time = setup()

    class DepartingRaider(DefenseGame):
        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            if any(s.kind == 'key' and s.keys == ('p',) for s in steps):
                self.ctx.last_canvas = frame('base-reactors.png')
                draw_bar(self.ctx.last_canvas, 300, 400, 36, 0.9)

    executor = runner.executor = DepartingRaider(ctx)
    runner.start('army:defend', 'army', defense_session(ctx, rounds=2, every=0.2, reach=600))
    for _ in range(30):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.busy('army'):
            break
    assert runner.finished[-1].status == 'done'
    assert not any(step[0] == 'click' and step[2] for _, _, act in executor.acts for step in act)
    assert next(e for e in world.events if e.get('order') == 'defense')['orders'] == 0


def test_quiet_base_gives_no_orders():
    world, ctx, runner, army, executor, time = setup()
    world.units_built['rhino_tank'] = 3
    ctx.last_canvas = frame('base-reactors.png')
    assert army.step(world.snapshot()) is None and not runner.busy('army')


@pytest.mark.parametrize('profile', ['startlux-ra2-allied.json', 'startlux-ra2-soviet.json'])
@pytest.mark.parametrize('act', [False, True])
def test_fresh_home_raid_preempts_long_attack_preparation(profile, act):
    from addons.chronodivide import playbook
    from addons.chronodivide.tasks import Poll

    world, ctx, runner, army, executor, time = setup()
    army.book = playbook.load(Path(__file__).parents[1] / 'profiles' / profile)
    ctx.camera = 'home'
    ctx.last_canvas = frame('enemy-near-base.png')
    def preparation():
        yield Poll(lambda s, v: False, 999)
    runner.start('army:attack', 'army', preparation(), priority=2)
    runner.tick(world.snapshot(), None)
    plan = army.step(world.snapshot(), act=act)
    assert plan[0] == 'army:defend'
    assert runner.tasks['army'].name == ('army:defend' if act else 'army:attack')
    if act:
        assert runner.finished[-1].reason == 'preempted by army:defend'


@pytest.mark.parametrize('waves', [0, 1])
def test_focused_first_wave_role_preparation_still_yields_to_a_home_radar_raid(monkeypatch, waves):
    from addons.chronodivide.intent import StandingIntent
    from addons.chronodivide.tasks import Poll

    world, ctx, runner, army, executor, time = setup()
    ctx.camera = 'away'
    army.intent = StandingIntent(focus='attack', posture='attack')
    army.home_threat = ((940, 110), 12)
    army.campaign.active, army.campaign.started = True, time[0]
    army.campaign.waves = waves
    monkeypatch.setattr(army, '_look', lambda now: None)

    def preparation():
        yield Poll(lambda s, v: False, 999)

    runner.start('army:attack', 'army', preparation(), priority=2)
    runner.tick(world.snapshot(), None)
    army.step(world.snapshot())
    assert runner.tasks['army'].name == ('army:defend' if waves == 0 else 'army:attack')


def test_raiders_side_by_side_are_still_targeted_one_by_one():
    # Live game 34: two raiders merged into one 194 px blob, above the building size, and the
    # defense session found nothing to fire at.
    import numpy as np
    from addons.chronodivide.army import battle_order, raider_target
    tactical = Layout.for_canvas(1024, 768).tactical
    canvas = np.full((768, 1024, 3), (40, 120, 90), np.uint8)
    blue = (200, 98, 26)                                  # the enemy's house color (BGR), on the measured ramp
    canvas[340:352, 130:142] = blue
    canvas[340:352, 147:159] = blue
    canvas[356:366, 138:150] = blue
    assert battle_order(canvas, tactical, (425, 368), 600, 0) is None           # the old rule: a "building"
    point = raider_target(canvas, tactical, (425, 368), 600)
    assert point is not None and 130 <= point[0] <= 159 and 340 <= point[1] <= 366
    assert raider_target(frame('base-reactors.png'), tactical, (425, 368), 600) is None


@pytest.mark.parametrize('profile', ['startlux-ra2-allied.json', 'startlux-ra2-soviet.json'])
def test_build_focus_regroups_remote_defenders_once_after_a_finished_raid(profile):
    from addons.chronodivide import playbook
    from addons.chronodivide.executor import Step
    from addons.chronodivide.tasks import Act

    world, ctx, runner, army, executor, time = setup()
    army.book = playbook.load(Path(__file__).parents[1] / 'profiles' / profile)
    world.units_built[army.book['units']['tank']] = 3
    ctx.last_canvas = frame('base-reactors.png')
    world.note('order', order='defense', orders=1)
    time[0] += 4
    remote = {'distant_tank', 'distant_anti_air'}

    class RegroupGame:
        def __init__(self):
            self.selected, self.saved, self.orders = set(), set(), []

        def act(self, name, lane, steps, commands=1):
            for step in steps:
                if step.kind == 'click' and step.button == 'right':
                    self.selected = set()
                elif step.kind == 'key' and step.keys == ('p',):
                    self.selected = remote.copy()
                elif step.kind == 'chord' and step.keys == ('ctrl', '1'):
                    self.saved = self.selected.copy()
                elif step.kind == 'key' and step.keys == ('1',):
                    self.selected = self.saved.copy()
                elif step.kind == 'click' and step.keys == ('ctrl', 'shift'):
                    self.orders.append(self.selected.copy())

    game = runner.executor = RegroupGame()

    def sidebar():
        while True:
            yield Act([Step.click(ctx.clear_point, 'right')], commands=0)

    assert army.step(world.snapshot())[0] == 'army:regroup'
    runner.start('sidebar:inspect', 'sidebar', sidebar(), priority=5)
    for _ in range(60):
        runner.tick(world.snapshot(), None)
        time[0] += 0.2
        if not runner.busy('army'):
            break
    assert not runner.busy('army')
    assert game.orders == [remote]
    assert army.step(world.snapshot()) != ('army:regroup', None)


@pytest.mark.parametrize('condition', ['threat', 'campaign', 'defend', 'no_order', 'zero_orders'])
def test_regroup_requires_completed_defense_and_a_quiet_gathering_phase(condition):
    from addons.chronodivide.intent import StandingIntent

    world, ctx, runner, army, executor, time = setup()
    army.intent = StandingIntent()
    world.units_built['rhino_tank'] = 3
    ctx.last_canvas = frame('base-reactors.png')
    if condition != 'no_order':
        world.note('order', order='defense', orders=0 if condition == 'zero_orders' else 1)
    time[0] += 4
    if condition == 'threat':
        army.threat_at = time[0]
    elif condition == 'campaign':
        army.campaign.active = True
    elif condition == 'defend':
        army.intent.posture = 'defend'
    plan = army.step(world.snapshot())
    assert plan is None or plan[0] != 'army:regroup'
