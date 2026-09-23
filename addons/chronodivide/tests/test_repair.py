from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.client import ClientProfile
from addons.chronodivide import playbook
from addons.chronodivide.executor import Step
from addons.chronodivide.frame_player import FramePlayer
from addons.chronodivide.perception.buildings import Building, default_recognizer
from addons.chronodivide.repair import mode_icon, selected_health
from addons.chronodivide.tasks import Act, Look


FIXTURES = Path(__file__).with_name('fixtures')


def fixture(name):
    return cv2.imread(str(FIXTURES / (name+'.png')))


def test_live_selected_health_and_cursor_mode():
    healthy = fixture('repair-healthy-browser')
    damaged = fixture('repair-damaged-refinery-browser')
    active = fixture('repair-mode-browser')
    layout = ClientProfile.load().layout()
    assert selected_health(healthy, (280, 210, 540, 320)) == 'green'
    assert selected_health(damaged, (370, 300, 550, 390)) == 'yellow'
    assert selected_health(damaged, (370, 300, 550, 390), before=damaged) is None
    assert mode_icon(healthy, layout, 'repair') == 'inactive'
    assert mode_icon(active, layout, 'repair') == 'active'
    assert mode_icon(active, layout, 'sell') == 'inactive'


def test_native_allied_cyan_repair_highlight_and_blue_exit_icons():
    layout = ClientProfile.load('chronodivide-allied').layout()
    active = fixture('allied-repair-icons-active')
    inactive = fixture('allied-repair-icons-inactive')
    assert mode_icon(active, layout, 'repair') == 'active'
    assert mode_icon(active, layout, 'sell') == 'inactive'
    assert mode_icon(inactive, layout, 'repair') == 'inactive'
    assert mode_icon(inactive, layout, 'sell') == 'inactive'
    assert mode_icon(np.zeros_like(active), layout, 'repair') is None
    assert mode_icon(np.zeros_like(active), layout, 'sell') is None


def test_horizontal_unit_bars_and_old_sprite_colors_are_not_damage():
    before = np.zeros((100, 150, 3), np.uint8)
    canvas = before.copy()
    cv2.line(canvas, (20, 70), (100, 70), (0, 240, 240), 3)
    assert selected_health(canvas, (0, 0, 150, 100), before) is None
    cv2.line(canvas, (20, 70), (100, 30), (0, 240, 240), 3)
    assert selected_health(canvas, (0, 0, 150, 100), before) == 'yellow'
    assert selected_health(canvas, (0, 0, 150, 100), canvas) is None


def test_live_burning_yard_short_red_bar_requires_selection_delta_and_calibrated_diagonal():
    damaged = fixture('repair-damaged-yard-browser')
    before = fixture('repair-damaged-yard-unselected-browser')
    # Short tips are accepted only along the calibrated yard bar after selection.
    assert selected_health(damaged, (300, 195, 530, 320)) is None
    diagonal = (311,294,433,235)
    assert selected_health(damaged, (300,195,530,320), before, 12, diagonal) == 'red'
    assert selected_health(damaged, (300,195,530,320), damaged, 12, diagonal) is None
    assert selected_health(damaged, (300,195,530,320), None, 12, diagonal) is None


def test_live_fire_without_selected_bar_is_not_damage():
    before = fixture('repair-damaged-yard-unselected-browser')
    damaged = fixture('repair-damaged-yard-browser')
    # Both images contain actual fire; only the selected image exposes red pips.
    assert selected_health(before, (300,195,530,320), damaged, 12, (311,294,433,235)) is None


def test_real_damaged_yard_can_reach_k_with_cash_but_never_with_zero_cash():
    player, repair = controller()
    before = fixture('repair-damaged-yard-unselected-browser')
    selected = fixture('repair-damaged-yard-browser')
    building = next(b for b in default_recognizer().detect(before) if b.name == 'construction_yard')
    for cash in (0, 1200):
        player.world.set('credits', cash, 10)
        player.ctx.last_canvas = before
        task = repair._repair(building, (428,344))
        next(task)
        baseline = next(task).fn(None, None)
        task.send(baseline)
        player.ctx.last_canvas = selected
        level = next(task).fn(None, None)
        assert level == 'red'
        if cash:
            assert task.send(level).steps[0].kind == 'repair_mode'
        else:
            with pytest.raises(StopIteration):
                task.send(level)
        task.close()


def test_native_damaged_factory_enters_repair_from_fresh_selected_health():
    player, repair = controller()
    before = fixture('soviet-factory-damaged-unselected')
    selected = fixture('soviet-factory-damaged-selected')
    building = default_recognizer().detect(before, (0,0,220,220))[0]
    assert building.name in repair.policy['items']
    player.ctx.last_canvas = before
    task = repair._repair(building, (108,136))
    next(task)
    baseline = next(task).fn(None, None)
    task.send(baseline)
    player.ctx.last_canvas = selected
    level = next(task).fn(None, None)
    assert level == 'yellow'
    # The compact recording omits the HUD, whose mode was verified separately.
    player.ctx.last_canvas = fixture('repair-healthy-browser')
    assert task.send(level).steps[0].kind == 'repair_mode'
    task.close()


def test_burning_allied_power_plant_reaches_repair_only_from_its_selected_pips():
    player = FramePlayer(profile=ClientProfile.load('chronodivide-allied'), compressed=True,
                         book=playbook.load(Path(__file__).resolve().parents[1] /
                                            'profiles/startlux-ra2-allied.json'), clock=lambda: 10)
    repair = player.repair
    player.world.set('credits', 1200, 10)
    before = fixture('allied-power-plant-damaged-before-selection')
    selected = fixture('allied-power-plant-damaged-selected')
    building = Building((25, 60, 100, 145), 'ours', 'power_plant', .9)
    region, diagonal = (0, 0, 100, 125), (4, 69, 74, 34)
    assert selected_health(selected, region, before) is None
    assert selected_health(fixture('allied-power-plant-damaged-heldout'), (15, 0, 115, 120),
                           fixture('allied-power-plant-damaged-unselected'), 10, (19, 64, 89, 29)) is None
    player.ctx.last_canvas = before
    task = repair._repair(building, (62, 102))
    next(task)
    baseline = next(task).fn(None, None)
    task.send(baseline)
    player.ctx.last_canvas = selected
    level = next(task).fn(None, None)
    assert level == 'yellow'
    player.ctx.last_canvas = fixture('allied-repair-icons-inactive')
    assert task.send(level).steps[0].kind == 'repair_mode'
    task.close()


def test_allied_factory_healthy_selected_bar_abstains_from_repair():
    player = FramePlayer(profile=ClientProfile.load('chronodivide-allied'), compressed=True,
                         book=playbook.load(Path(__file__).resolve().parents[1] /
                                            'profiles/startlux-ra2-allied.json'), clock=lambda: 10)
    repair = player.repair
    before = fixture('allied-factory-healthy')
    selected = fixture('allied-factory-healthy-selected')
    building = default_recognizer().detect(before, compressed=True)[0]
    assert repair.policy['items'][0] == building.name == 'war_factory'
    player.world.set('credits', 5000, 10)
    player.ctx.last_canvas = before
    task = repair._repair(building, (113,123))
    next(task)
    baseline = next(task).fn(None, None)
    task.send(baseline)
    player.ctx.last_canvas = selected
    level = next(task).fn(None, None)
    assert level == 'green'
    with pytest.raises(StopIteration):
        task.send(level)
    assert not repair.latched


def controller():
    player = FramePlayer(clock=lambda: 10)
    player.ctx.last_canvas = fixture('repair-healthy-browser')
    player.world.set('credits', 5000, 10)
    return player, player.repair


@pytest.mark.parametrize('cash,enters', [(299, False), (300, True), (675, True)])
def test_allied_damage_repair_can_start_without_waiting_for_tank_sized_savings(cash, enters):
    book = playbook.load(Path(__file__).parents[1] / 'profiles' / 'startlux-ra2-allied.json')
    player = FramePlayer(clock=lambda: 10, profile=ClientProfile.load('chronodivide-allied'), book=book)
    player.ctx.last_canvas = fixture('allied-repair-icons-inactive')
    player.world.set('credits', cash, 10)
    task = player.repair._repair(Building((325,255,530,432), 'ours', 'war_factory', .9), (428,344))
    next(task); next(task)
    task.send(player.ctx.last_canvas.copy()); next(task)
    if enters:
        assert task.send('yellow').steps[0].kind == 'repair_mode'
        task.close()
    else:
        with pytest.raises(StopIteration):
            task.send('yellow')


def test_fresh_damage_enters_mode_once_and_right_click_cleans_up(monkeypatch):
    player, repair = controller()
    monkeypatch.setattr('addons.chronodivide.repair.selected_health', lambda *a: 'yellow')
    building = Building((325, 255, 530, 432), 'ours', 'construction_yard', .9)
    task = repair._repair(building, (428, 344))
    assert isinstance(next(task), Act)
    assert isinstance(next(task), Look)
    assert isinstance(task.send(player.ctx.last_canvas.copy()), Act)
    assert isinstance(next(task), Look)
    enter = task.send('yellow')
    assert enter.steps[0].kind == 'repair_mode'
    assert repair.cleanup_pending
    assert isinstance(next(task), Look)
    click = task.send(True)
    assert click.steps[0].point == (428, 344)
    assert repair.latched == [(428, 344)]
    cleanup = next(task)
    assert cleanup.steps[0].button == 'right'
    assert isinstance(next(task), Look)
    with pytest.raises(StopIteration):
        task.send(True)
    assert not repair.cleanup_pending


@pytest.mark.parametrize('level,cash', [('green', 5000), ('yellow', 1000), (None, 5000)])
def test_healthy_unknown_or_insufficient_reserve_never_enters_repair(level, cash):
    player, repair = controller()
    player.world.set('credits', cash, 10)
    task = repair._repair(Building((325,255,530,432), 'ours', 'construction_yard', .9), (428,344))
    next(task); next(task)
    task.send(player.ctx.last_canvas.copy()); next(task)
    with pytest.raises(StopIteration):
        task.send(level)
    assert not repair.cleanup_pending and not repair.latched


def test_cancellation_after_k_keeps_cleanup_guard_and_blocks_planners():
    player, repair = controller()
    task = repair._repair(Building((325,255,530,432), 'ours', 'construction_yard', .9), (428,344))
    next(task); next(task)
    task.send(player.ctx.last_canvas.copy()); next(task)
    assert task.send('yellow').steps[0].kind == 'repair_mode'
    task.close()
    assert repair.cleanup_pending
    assert repair.step(player.world.snapshot())
    assert player.runner.tasks['sidebar'].name == 'repair:cleanup'


def test_browser_repair_key_requires_dedicated_task():
    player, _ = controller()
    with pytest.raises(ValueError):
        player.act('queue:tank', 'sidebar', [Step.repair_mode()])
    player.act('repair:tesla_reactor', 'sidebar', [Step.repair_mode()])
    assert player.pending['steps'][0]['kind'] == 'key'
    assert player.pending['steps'][0]['keys'] == ('k',)


def test_scheduler_respects_reserves_placement_and_once_per_building_latch(monkeypatch):
    player, repair = controller()
    building = Building((325,255,530,432), 'ours', 'construction_yard', .9)
    monkeypatch.setattr('addons.chronodivide.repair.default_recognizer',
                        lambda: type('Recognizer', (), {'detect': lambda _, canvas, **kw: [building]})())
    player.world.set('credits', 1199, 10)
    assert not repair.step(player.world.snapshot())
    repair.next_scan = 0
    player.world.set('credits', 5000, 10)
    snapshot = player.world.snapshot()
    snapshot['lanes']['buildings']['status'] = 'ready'
    assert not repair.step(snapshot)
    snapshot['lanes']['buildings']['status'] = 'idle'
    repair.latched.append((428,344))
    assert not repair.step(snapshot)
    repair.next_scan = 0
    repair.latched.clear()
    assert repair.step(snapshot)
    assert player.runner.tasks['sidebar'].name == 'repair:construction_yard'


def test_unknown_cleanup_icon_keeps_input_blocked():
    player, repair = controller()
    repair.cleanup_pending = True
    task = repair._cleanup()
    next(task); next(task)
    with pytest.raises(StopIteration):
        task.send(False)
    assert repair.cleanup_pending
    repair.cleanup_attempts = repair.policy['cleanup_attempts']
    assert repair.step(player.world.snapshot())
    assert not player.runner.tasks


def test_slow_health_scan_waits_a_full_interval_after_completion(monkeypatch):
    player, repair = controller()
    now = [10]
    repair.clock = lambda: now[0]
    player.clock = player.world.clock = player.runner.clock = repair.clock
    building = Building((325,255,530,432), 'ours', 'construction_yard', .9)
    monkeypatch.setattr('addons.chronodivide.repair.default_recognizer',
                        lambda: type('Recognizer', (), {'detect': lambda _, canvas, **kw: [building]})())
    monkeypatch.setattr(repair, '_repair', lambda *args: (step for step in ()))
    assert repair.step(player.world.snapshot())
    now[0] += repair.policy['scan_every'] + 5  # transport outlasted the original scan interval
    assert repair.step(player.world.snapshot())
    player.runner.tick(player.world.snapshot(), None)
    assert not repair.step(player.world.snapshot())
    next_scan = repair.next_scan
    assert next_scan == now[0] + repair.policy['scan_every']
    now[0] = next_scan - 1
    player.world.set('credits', 5000, now[0])
    assert not repair.step(player.world.snapshot())
    now[0] = next_scan
    player.world.set('credits', 5000, now[0])
    assert repair.step(player.world.snapshot())


@pytest.mark.parametrize('lane,allowed', [('buildings', False), ('defenses', False), ('vehicles', True)])
def test_finished_unit_at_factory_exit_does_not_block_building_repair(monkeypatch, lane, allowed):
    player, repair = controller()
    building = Building((88,369,269,524), 'ours', 'war_factory', .9)
    monkeypatch.setattr('addons.chronodivide.repair.default_recognizer',
                        lambda: type('Recognizer', (), {'detect': lambda _, canvas, **kw: [building]})())
    snapshot = player.world.snapshot()
    snapshot['lanes'][lane]['status'] = 'ready'
    assert repair.step(snapshot) == allowed


def test_repair_rotation_starts_with_profile_priority_and_excludes_enemy_buildings(monkeypatch):
    player, repair = controller()
    buildings = [Building((325,255,530,432), 'ours', 'construction_yard', .9),
                 Building((88,219,269,374), 'enemy', 'war_factory', .9),
                 Building((88,369,269,524), 'ours', 'war_factory', .9)]
    monkeypatch.setattr('addons.chronodivide.repair.default_recognizer',
                        lambda: type('Recognizer', (), {'detect': lambda _, canvas, **kw: buildings})())
    assert repair.step(player.world.snapshot())
    assert player.runner.tasks['sidebar'].name == 'repair:war_factory'


@pytest.mark.parametrize('name,lane,allowed', [('look:vehicles', 'sidebar', True),
                                            ('queue:rhino_tank', 'sidebar', False),
                                            ('place:war_factory', 'sidebar', False),
                                            ('army:attack', 'army', False)])
def test_repair_can_interrupt_observation_but_not_production_placement_or_combat(monkeypatch, name, lane, allowed):
    player, repair = controller()
    building = Building((88,219,269,374), 'ours', 'war_factory', .9)
    monkeypatch.setattr('addons.chronodivide.repair.default_recognizer',
                        lambda: type('Recognizer', (), {'detect': lambda _, canvas, **kw: [building]})())
    player.runner.start(name, lane, iter(()), priority=3)
    assert bool(repair.step(player.world.snapshot())) == allowed
    if allowed:
        assert player.runner.tasks['sidebar'].name == 'repair:war_factory'
    else:
        assert player.runner.tasks[lane].name == name
