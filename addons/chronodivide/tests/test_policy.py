"""Reflex policy and skill coroutines against a scripted world (no screen, no input)."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from addons.chronodivide.army import Army
from addons.chronodivide.executor import Trust
from addons.chronodivide.intent import StandingIntent
from addons.chronodivide.layout import Layout
from addons.chronodivide.reflex import Reflex
from addons.chronodivide import skills
from addons.chronodivide.tasks import TaskRunner
from addons.chronodivide.world import Slot, World


class Recorder:
    def __init__(self):
        self.acts = []

    def act(self, name, lane, steps, commands=1):
        self.acts.append((name, [(s.kind, s.point, s.keys, s.delta) for s in steps]))


def setup(clock=None):
    time = clock or [100.0]
    world = World(clock=lambda: time[0])
    layout = Layout.for_canvas(1024, 768)
    ctx = skills.Context(layout, world)
    executor = Recorder()
    runner = TaskRunner(executor, Trust(), clock=lambda: time[0])
    intent = StandingIntent()
    army = Army(ctx, runner, intent, clock=lambda: time[0])
    reflex = Reflex(ctx, runner, None, intent, clock=lambda: time[0], army=army)
    runner.on_failure = reflex.failed
    return world, ctx, runner, reflex, executor, time


def run(runner, world, time, until=lambda: False, ticks=12, step=0.2):
    for _ in range(ticks):
        runner.tick(world.snapshot(), None)
        if until():
            return True
        time[0] += step
    return until()


def show(world, lane, *slots):
    world.update_sidebar(lane, [Slot(i, item, state, progress) for i, (item, state, progress) in enumerate(slots)])


def test_finishing_building_does_not_queue_again_before_ready_text():
    world, ctx, runner, reflex, executor, time = setup()
    world.update_sidebar('buildings', [Slot(0, 'tesla_reactor', 'building', .97)])
    time[0] += .7
    world.update_sidebar('buildings', [Slot(0, 'tesla_reactor', 'available')])
    assert world.lanes['buildings'].status == 'building'
    time[0] += .7
    world.update_sidebar('buildings', [Slot(0, 'tesla_reactor', 'ready', 1)])
    assert world.lanes['buildings'].status == 'ready'


def test_cancelled_building_releases_busy_memory_after_two_seconds():
    world, ctx, runner, reflex, executor, time = setup()
    world.update_sidebar('buildings', [Slot(0, 'tesla_reactor', 'building', .4)])
    for delay in (.7, .7, .7):
        time[0] += delay
        world.update_sidebar('buildings', [Slot(0, 'tesla_reactor', 'available')])
    assert world.lanes['buildings'].status == 'idle'


def test_first_anti_air_precedes_expansion_and_saves_only_its_cost():
    world, ctx, runner, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, ore_refinery=1, war_factory=1)
    world.units_built['rhino_tank'] = reflex.book['units']['second_refinery']['after_tanks']
    available = ['war_miner', 'rhino_tank', 'flak_track']
    for credits in (500, 1400, 4000):
        world.set('credits', credits)
        assert reflex.next_item('vehicles', available, time[0]) == 'flak_track'
    world.set('credits', 499)
    assert reflex.next_item('vehicles', available, time[0]) is None
    world.units_built['flak_track'] = 1
    world.set('credits', 1400)
    assert reflex.next_item('vehicles', available, time[0]) is None  # second refinery savings resume
    world.units_built['rhino_tank'] = reflex.book['units']['anti_air']['after_tanks'] - 1
    world.units_built['flak_track'] = 0
    assert reflex.next_item('vehicles', available, time[0]) == 'rhino_tank'


def test_scout_quota_waits_for_dog_cameo_before_other_infantry():
    world, ctx, runner, reflex, executor, time = setup()
    plan = reflex.book['units']['scouts']
    world.units_built[plan['item']] = plan['count'] - 1
    # The finishing dog's cameo can briefly be unreadable while Conscripts remain available.
    assert reflex.unit_options('infantry', ['conscript'], 5000, time[0])[0] == []
    assert reflex.unit_options('infantry', ['conscript', plan['item']], 5000, time[0])[0] == [plan['item']]
    world.units_built[plan['item']] = plan['count']
    reflex.army.scouts_ready = True
    assert reflex.unit_options('infantry', ['conscript', plan['item']], 5000, time[0])[0] == ['conscript']


FIXTURES = Path(__file__).with_name('fixtures')


def test_first_action_deploys_the_mcv_by_keys_before_any_calibration():
    world, ctx, runner, reflex, executor, time = setup()
    ctx.pointer = SimpleNamespace(name='locked', position=None)          # not calibrated yet
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'frame-start.png'))     # MCV before deployment
    world.set('sidebar_empty', True)
    assert reflex.step(world.snapshot())[0] == 'deploy_mcv'
    runner.tick(world.snapshot(), None)
    assert [s[2] for s in executor.acts[0][1] if s[0] == 'key'] == [('n',), ('s',), ('d',)]
    world.set('sidebar_empty', False)
    show(world, 'buildings', ('tesla_reactor', 'available', None))
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'frame-play.png'))      # base visible after H
    assert run(runner, world, time, lambda: not runner.busy('sidebar'))
    assert runner.finished[-1].status == 'done'
    assert world.structures['construction_yard'] == 1 and world.mcv_deployed.value
    assert executor.acts[-1][1][0][2] == ('h',) and ctx.home_known
    assert reflex.step(world.snapshot(), act=False)[0] == 'calibrate_pointer'


def test_mcv_blocked_where_it_stands_drives_to_open_ground_and_deploys_there():
    world, ctx, runner, reflex, executor, time = setup()
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'frame-start.png'))
    world.set('sidebar_empty', True)
    runner.start('deploy_mcv', 'sidebar', skills.deploy_mcv(ctx, keys=False))
    clicks = lambda: [s[1] for a in executor.acts for s in a[1] if s[0] == 'click']
    assert run(runner, world, time, lambda: len(clicks()) == 3, ticks=40)
    select, deploy, move = clicks()
    assert select == deploy and abs(select[0] - 430) <= 8 and abs(select[1] - 358) <= 8
    assert move != deploy                                  # blocked: a move order to open ground
    assert any(e['kind'] == 'mcv-moved' for e in world.events) or runner.busy('sidebar')


def test_mid_game_attach_never_tries_to_deploy():
    world, ctx, runner, reflex, executor, time = setup()
    show(world, 'infantry', ('conscript', 'available', None))
    world.set('sidebar_empty', False)
    plan = reflex.step(world.snapshot(), act=False)
    assert plan is None or plan[0] != 'deploy_mcv'


def test_idle_buildings_lane_queues_the_build_order_by_clicking_the_right_slot():
    world, ctx, runner, reflex, executor, time = setup()
    world.structures['construction_yard'] = 1
    world.set('credits', 10000)
    show(world, 'buildings', ('tesla_reactor', 'available', None))
    assert reflex.step(world.snapshot())[0] == 'queue:tesla_reactor'
    runner.tick(world.snapshot(), None)
    assert executor.acts[-1][1][0][1] == ctx.slot_point(0)
    show(world, 'buildings', ('tesla_reactor', 'building', 0.1))
    assert run(runner, world, time, lambda: not runner.busy('sidebar'), ticks=2)
    assert runner.finished[-1].status == 'done' and world.ledger.reserved() == 0


GHOST_GREEN, GHOST_RED = (23, 220, 51), (17, 38, 213)


class PlacementGame(Recorder):
    """Draws the footprint ghost where the cursor moves: one verdict per probe ('red' or 'green').

    `holding` is whether the key pick-up worked; if not, the ghost appears only after
    the Ready cameo is clicked.
    """

    def __init__(self, ctx, verdicts, holding=True):
        super().__init__()
        self.ctx, self.verdicts, self.holding = ctx, list(verdicts), holding
        self.base = cv2.imread(str(FIXTURES / 'frame-play.png'))
        ctx.last_canvas = self.base.copy()

    def act(self, name, lane, steps, commands=1):
        super().act(name, lane, steps, commands)
        frame = self.base.copy()
        for step in steps:
            if step.kind == 'click' and self.ctx.layout.grid.box(0).contains(step.point):
                self.holding = True
            if step.kind == 'move' and step.point != self.ctx.park and self.holding and self.verdicts:
                x, y = step.point
                frame[y - 30:y + 30, x - 60:x + 60] = GHOST_GREEN
                if self.verdicts.pop(0) == 'red':
                    frame[y - 30:y + 30, x + 20:x + 60] = GHOST_RED       # blocked on the right side
        self.ctx.last_canvas = frame


def test_ready_building_is_placed_where_its_footprint_is_all_green():
    world, ctx, runner, reflex, executor, time = setup()
    executor = runner.executor = PlacementGame(ctx, ['red', 'green'])
    world.structures['construction_yard'] = 1
    show(world, 'buildings', ('tesla_reactor', 'ready', 1.0), ('ore_refinery', 'disabled', None))
    assert reflex.step(world.snapshot())[0] == 'place:tesla_reactor'
    assert run(runner, world, time, lambda: any(s[0] == 'click' for a in executor.acts for s in a[1]), ticks=30)
    show(world, 'buildings', ('tesla_reactor', 'available', None), ('ore_refinery', 'available', None))
    assert run(runner, world, time, lambda: not runner.busy('sidebar'))
    assert runner.finished[-1].status == 'done' and world.structures['tesla_reactor'] == 1
    steps = [s for a in executor.acts for s in a[1]]
    assert steps[0][2] == ('h',) and steps[1][2] == ('q',)       # center the base, pick up with Q
    probes = [s[1] for s in steps if s[0] == 'move' and s[1] != ctx.park]
    clicks = [s[1] for s in steps if s[0] == 'click']
    assert len(probes) == 2 and clicks == [probes[1]]            # the blocked spot is never clicked
    assert probes[1] == (probes[0][0] - 60, probes[0][1])        # one cell away from the red side


def test_placement_clicks_the_ready_cameo_when_the_key_did_not_pick_up_the_building():
    world, ctx, runner, reflex, executor, time = setup()
    executor = runner.executor = PlacementGame(ctx, ['green'], holding=False)
    world.structures['construction_yard'] = 1
    show(world, 'buildings', ('tesla_reactor', 'ready', 1.0))
    reflex.step(world.snapshot())
    assert run(runner, world, time, lambda: sum(s[0] == 'click' for a in executor.acts for s in a[1]) == 2,
               ticks=30)
    clicks = [s[1] for a in executor.acts for s in a[1] if s[0] == 'click']
    assert clicks[0] == ctx.slot_point(0)                         # no footprint seen: cameo, then the spot


def test_disabled_defense_tab_after_yard_loss_never_clicks_infantry_cameo():
    world, ctx, runner, reflex, executor, time = setup()
    ctx.last_canvas = np.full((768, 1024, 3), 100, np.uint8)
    show(world, 'defenses', ('soviet_wall', 'available', None), ('sentry_gun', 'ready', 1.0))
    world.structures['construction_yard'] = 0
    show(world, 'infantry', ('conscript', 'available', None), ('engineer', 'available', None))
    assert ctx.slot_index('sentry_gun', ('ready',)) is None
    runner.start('place:sentry_gun', 'sidebar', skills.place_building(ctx, 'sentry_gun'))
    assert run(runner, world, time, lambda: not runner.busy('sidebar'), ticks=30)
    assert runner.finished[-1].status == 'failed'
    assert runner.finished[-1].reason == 'timeout waiting: defenses tab'
    assert not [s for a in executor.acts for s in a[1] if s[0] == 'click']
    assert world.units_built.get('engineer', 0) == 0


def test_placement_stops_if_sidebar_changes_after_successful_tab_selection():
    world, ctx, runner, reflex, executor, time = setup()
    executor = runner.executor = PlacementGame(ctx, ['green'], holding=False)
    show(world, 'defenses', ('soviet_wall', 'available', None), ('sentry_gun', 'ready', 1.0))
    runner.start('place:sentry_gun', 'sidebar', skills.place_building(ctx, 'sentry_gun'))
    probes = lambda: [s for a in executor.acts for s in a[1] if s[0] == 'move' and s[1] != ctx.park]
    assert run(runner, world, time, lambda: bool(probes()), ticks=20)
    show(world, 'infantry', ('conscript', 'available', None), ('engineer', 'available', None))
    assert run(runner, world, time, lambda: not runner.busy('sidebar'), ticks=20)
    assert runner.finished[-1].status == 'failed' and 'no longer active' in runner.finished[-1].reason
    assert not [s for a in executor.acts for s in a[1] if s[0] == 'click']


def test_tabs_switch_by_key_and_fall_back_to_clicking():
    world, ctx, runner, reflex, executor, time = setup()
    show(world, 'buildings', ('tesla_reactor', 'available', None))
    runner.start('look', 'sidebar', skills.select_tab(ctx, 'infantry'))
    runner.tick(world.snapshot(), None)
    assert executor.acts[-1][1][0][2] == ('e',)
    show(world, 'infantry', ('conscript', 'available', None))
    assert run(runner, world, time, lambda: not runner.busy('sidebar'))
    for _ in range(3):                                   # the key does not work: fall back to clicking
        runner.start('look', 'sidebar', skills.select_tab(ctx, 'vehicles'))
        run(runner, world, time, lambda: executor.acts[-1][1][0][0] == 'click')
        show(world, 'vehicles', ('war_miner', 'available', None))
        run(runner, world, time, lambda: not runner.busy('sidebar'))
        show(world, 'infantry', ('conscript', 'available', None))
    assert ctx.keyboard_tabs is True and ctx.keyboard_tab_misses == 3    # keys are tried again next time


def test_structure_tab_is_clicked_so_a_finished_building_is_not_picked_up():
    world, ctx, runner, reflex, executor, time = setup()
    show(world, 'infantry', ('conscript', 'available', None))
    runner.start('look', 'sidebar', skills.select_tab(ctx, 'buildings'))
    runner.tick(world.snapshot(), None)
    assert executor.acts[-1][1][0][:2] == ('click', ctx.tab_point('buildings'))


def test_visible_idle_lane_is_served_before_refreshing_stale_tabs():
    world, ctx, runner, reflex, executor, time = setup()
    scouts = reflex.book['units']['scouts']
    world.units_built[scouts['item']] = scouts['count']
    reflex.army.scouts_ready = True
    world.structures.update(construction_yard=1, barracks=1, tesla_reactor=1, ore_refinery=1)
    world.set('credits', 5000)
    show(world, 'buildings', ('tesla_reactor', 'building', 0.4))
    time[0] += 5
    show(world, 'infantry', ('conscript', 'available', None), ('engineer', 'available', None))
    assert reflex.step(world.snapshot(), act=False)[0] == 'queue:conscript'
    show(world, 'infantry', ('conscript', 'building', 0.2), ('engineer', 'disabled', None))
    # Defenses are not wanted (no enemy has come near the base), so the stale buildings tab is refreshed.
    assert reflex.step(world.snapshot(), act=False)[0] == 'look:buildings'
    reflex.army.threat_at = time[0]                  # enemies near the base: defenses become wanted
    assert reflex.step(world.snapshot(), act=False)[0] == 'look:defenses'   # never seen: stalest


def test_low_power_builds_a_reactor_first():
    world, ctx, runner, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, tesla_reactor=1, ore_refinery=1, barracks=1)
    world.set('credits', 5000)
    world.set('power', dict(level='low', fill=0.3))
    show(world, 'buildings', ('tesla_reactor', 'available', None), ('war_factory', 'available', None))
    assert reflex.step(world.snapshot(), act=False)[0] == 'queue:tesla_reactor'


def test_strategy_only_offers_feasible_objectives_and_postures():
    world, ctx, runner, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, ore_refinery=1)
    assert set(reflex.feasible_objectives()) == {'build_army', 'expand_economy'}
    assert set(reflex.feasible_postures(time[0])) == {'gather'}
    world.structures.update(war_factory=1)
    below_default = reflex.book.option('attack_size', 'large')['tanks'] - 1
    world.units_built.update(rhino_tank=below_default)
    assert 'tech_up' not in reflex.feasible_objectives()          # production totals do not prove an advantage
    assert 'attack' not in reflex.feasible_postures(time[0])        # radar offline: nowhere to go
    world.set('radar', SimpleNamespace(online=True, enemy=(), own=()))
    assert 'attack' not in reflex.feasible_postures(time[0])       # one tank below the rule's first attack
    reflex.intent.set('attack_size', below_default, 'systemone')        # System One accepts the force already at home
    assert 'attack' in reflex.feasible_postures(time[0])
    reflex.army.threat_at = time[0] - 5
    assert 'defend' in reflex.feasible_postures(time[0])


def calibration(positions):
    world, ctx, runner, reflex, executor, time = setup()
    found = iter(positions)
    ctx.cursor_finder = SimpleNamespace(find=lambda canvas, region=None: next(found))
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    ctx.pointer = SimpleNamespace(name='locked', position=None, gain=None,
                                  observe=lambda p: setattr(ctx.pointer, 'position', p))
    runner.start('calibrate', 'sidebar', skills.calibrate_pointer(ctx))
    run(runner, world, time, lambda: not runner.busy('sidebar'), ticks=20)
    return ctx, runner, executor


def test_pointer_calibration_pins_the_cursor_over_the_sidebar_then_learns_gain():
    # After pinning in the top right corner and stepping back, the cursor is seen at (953, 250),
    # over the cameo grid whatever its height was (over the online radar it is drawn small).
    ctx, runner, executor = calibration([(953, 250), (953, 510), (928, 510)])
    assert runner.finished[-1].status == 'done', runner.finished[-1].reason
    assert ctx.pointer.gain == 1.0 and ctx.pointer.position == (928, 510)
    nudges = [step[3] for act in executor.acts for step in act[1] if step[0] == 'nudge']
    assert nudges == [(1100, -1100), (1100, -1100), (-70, 250), (0, 260), (-25, 0)]
    assert executor.acts[-1][1][0][:2] == ('move', ctx.park)          # ends parked over the radar


def test_nonlinear_pointer_motion_stops_calibration_with_advice():
    ctx, runner, executor = calibration([(953, 250), (953, 650), (928, 650)])   # 260 px moved 400: acceleration
    assert runner.finished[-1].status == 'failed' and 'acceleration' in runner.finished[-1].reason


def test_queue_click_parks_the_cursor_off_the_icons():
    world, ctx, runner, reflex, executor, time = setup()
    world.structures['construction_yard'] = 1
    world.set('credits', 10000)
    show(world, 'buildings', ('tesla_reactor', 'available', None))
    reflex.step(world.snapshot())
    runner.tick(world.snapshot(), None)
    steps = executor.acts[-1][1]
    assert steps[0][0] == 'click' and steps[1] == ('move', ctx.park, (), None)


def test_build_order_waits_for_the_next_item_but_not_forever():
    world, ctx, runner, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, tesla_reactor=1, ore_refinery=1, barracks=1)
    world.set('credits', 10000)
    world.units_built['rhino_tank'] = 4                                     # the radar is due
    available = ['tesla_reactor', 'radar_tower', 'ore_refinery']
    assert reflex.next_item('buildings', available, time[0]) is None      # war factory not shown yet: wait
    assert reflex.next_item('buildings', available, time[0] + 25) == 'ore_refinery'  # skipped ahead to expansion
