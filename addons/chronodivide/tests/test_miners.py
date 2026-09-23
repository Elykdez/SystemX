"""Temporary miner defense must not hijack ordinary economy or guess a return target."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide import miners
from addons.chronodivide.perception.resources import ore_target
from addons.chronodivide.perception.unitboxes import Unit
from addons.chronodivide.tasks import Act, Look, Poll
from addons.chronodivide.tests.test_playbook import game

FIXTURES = Path(__file__).with_name('fixtures')


@pytest.mark.parametrize('quality', [None, 90])
def test_native_war_miner_tooltip_is_required_for_soviet_recovery(quality):
    from addons.chronodivide.perception.unit_labels import read_label
    canvas = cv2.imread(str(FIXTURES / 'war-miner-label-browser.png'))
    if quality is not None:
        ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        canvas = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert read_label(canvas, (346, 319)) == 'war_miner'
    for point in ((100, 100), (700, 650)):
        assert read_label(canvas, point) is None


def vehicle(health=1):
    return Unit((390, 390, 420, 420), 'ours', 'vehicle', health)


def drive(task, ctx, time, limit=80):
    actions, value = [], None
    for _ in range(limit):
        try:
            step = task.send(value)
        except StopIteration:
            return actions
        value = None
        if isinstance(step, Act):
            actions.extend(step.steps)
        elif isinstance(step, Look):
            value = step.fn(ctx.world.snapshot(), None)
        elif isinstance(step, Poll):
            time[0] += step.timeout
            value = False
        else:
            raise AssertionError(type(step))
    raise AssertionError('Task did not terminate')


def support_setup(monkeypatch, selected=None):
    world, army, reflex, time = game()
    ctx = army.ctx
    ctx.compressed_frames, ctx.camera = True, 'home'
    state = army.miners
    state.grouped = True
    selections = iter(selected if selected is not None else [[vehicle()], [vehicle()]])
    monkeypatch.setattr(miners, 'selected', lambda ctx: next(selections))
    monkeypatch.setattr(miners, 'ground_target', lambda *args: (500, 400))
    monkeypatch.setattr(miners, 'ore_target', lambda *args: (600, 600))
    return ctx, state, army.book['defense']['emergency_miners'], time, army


def test_bootstrap_only_while_first_refinery_supplies_the_only_vehicle():
    world, army, reflex, time = game()
    plan = army.book['defense']['emergency_miners']
    world.set('mcv_deployed', True)
    world.structures['ore_refinery'] = 1
    assert miners.can_group(world, army.miners, plan)
    world.units_built['rhino_tank'] = 1
    assert not miners.can_group(world, army.miners, plan)
    world.units_built.clear()
    world.structures['war_factory'] = 1
    assert not miners.can_group(world, army.miners, plan)


@pytest.mark.parametrize('armed,expected', [(True, 'war_miner'), (False, 'chrono_miner')])
@pytest.mark.parametrize('label', ['war_miner', 'chrono_miner', 'prism_tank', None])
def test_late_miner_recovery_requires_native_identity_and_is_bounded(monkeypatch, label, armed, expected):
    from dataclasses import replace
    ctx, state, plan, time, army = support_setup(monkeypatch)
    plan = dict(plan, armed=armed)
    state.grouped = False
    unit = replace(vehicle(), bar=(390, 380, 36))
    monkeypatch.setattr(miners, 'selected', lambda ctx: [unit])
    monkeypatch.setattr(miners, 'read_label', lambda *args: label)
    actions = drive(miners.recover_group(ctx, state, plan), ctx, time, limit=300)
    assert state.grouped == (label == expected)
    assert sum(s.kind == 'key' and s.keys == ('n',) for s in actions) <= 32
    assert any(s.kind == 'chord' for s in actions) == (label == expected)
    assert not any(s.kind == 'click' and s.button == 'left' for s in actions)


@pytest.mark.parametrize('faction', ['allied', 'soviet'])
def test_late_miner_recovery_waits_for_lost_group_and_stalled_quiet_economy(faction):
    from addons.chronodivide import playbook
    world, army, reflex, time = game()
    army.book = playbook.load(f'addons/chronodivide/profiles/startlux-ra2-{faction}.json')
    world.set('credits', 500)
    army.miners.funding_at = time[0] - 100
    army.miners.income_at = time[0]
    assert miners.can_recover_group(army, time[0])
    army.miners.grouped = True
    assert not miners.can_recover_group(army, time[0])
    army.miners.grouped = False
    army.threat_at = time[0]
    assert not miners.can_recover_group(army, time[0])
    army.threat_at = -1e9
    army.campaign.active = True
    assert not miners.can_recover_group(army, time[0])
    army.campaign.active = False
    army.miners.attempts = army.book['defense']['emergency_miners']['group_attempts']
    assert not miners.can_recover_group(army, time[0])


def test_miner_recovery_releases_the_camera_when_income_resumes(monkeypatch):
    ctx, state, plan, time, army = support_setup(monkeypatch)
    state.grouped, state.funding_at = False, 10

    def funded(ctx):
        state.funding_at = 20
        return [vehicle()]

    monkeypatch.setattr(miners, 'selected', funded)
    monkeypatch.setattr(miners, 'read_label', lambda *args: pytest.fail('obsolete tooltip search'))
    actions = drive(miners.recover_group(ctx, state, plan), ctx, time)
    assert sum(s.kind == 'key' and s.keys == ('n',) for s in actions) == 1
    assert not state.grouped and ctx.camera == 'home'
    assert not any(s.kind == 'chord' for s in actions)


@pytest.mark.parametrize('reselected', [[], [vehicle(), vehicle()]])
def test_uncertain_team_reselection_never_establishes_miner_identity(monkeypatch, reselected):
    ctx, state, plan, time, army = support_setup(monkeypatch, [[vehicle()], reselected])
    state.grouped = False
    actions = drive(miners.group(ctx, state, plan), ctx, time)
    assert not state.grouped
    assert not any(s.kind == 'click' and s.button == 'left' for s in actions)


def test_early_rush_can_use_a_verified_miner_with_weak_defenders(monkeypatch):
    ctx, state, plan, time, army = support_setup(monkeypatch)
    actions = drive(miners.support(ctx, state, plan, lambda: time[0]), ctx, time)
    assert state.engaged
    assert any(s.point == (500, 400) and s.keys == ('ctrl', 'shift') for s in actions)


@pytest.mark.parametrize('reason', ['defenders', 'air_only', 'ore_missing', 'health', 'selection'])
def test_routine_or_unverified_cases_leave_miner_working(monkeypatch, reason):
    selections = [[vehicle()], [vehicle()]]
    if reason == 'defenders':
        selections = [[vehicle()] * 3]
    elif reason == 'health':
        selections[-1] = [vehicle(0.4)]
    elif reason == 'selection':
        selections[-1] = []
    ctx, state, plan, time, army = support_setup(monkeypatch, selections)
    if reason == 'air_only':
        monkeypatch.setattr(miners, 'ground_target', lambda *args: None)
    elif reason == 'ore_missing':
        monkeypatch.setattr(miners, 'ore_target', lambda *args: None)
    actions = drive(miners.support(ctx, state, plan, lambda: time[0]), ctx, time)
    assert not state.engaged
    assert not any(s.kind == 'click' and s.button == 'left' for s in actions)


@pytest.mark.parametrize('reason', ['timeout', 'health', 'raid_over'])
def test_emergency_exit_issues_harvest_order_only_at_fresh_ore(monkeypatch, reason):
    ctx, state, plan, time, army = support_setup(monkeypatch, [[vehicle(0.3 if reason == 'health' else 1)]])
    state.engaged, state.started = True, time[0]
    if reason == 'timeout':
        state.started -= plan['max_seconds']
    elif reason == 'raid_over':
        monkeypatch.setattr(miners, 'ground_target', lambda *args: None)
    actions = drive(miners.support(ctx, state, plan, lambda: time[0]), ctx, time)
    assert not state.engaged
    assert any(s.point == (600, 600) and not s.keys for s in actions)
    assert not any(s.keys == ('ctrl', 'shift') for s in actions)


def test_depleted_ore_stops_support_and_preserves_return_retry(monkeypatch):
    ctx, state, plan, time, army = support_setup(monkeypatch, [[vehicle()], [vehicle()]])
    state.engaged, state.started = True, time[0] - plan['max_seconds']
    monkeypatch.setattr(miners, 'ore_target', lambda *args: None)
    actions = drive(miners.support(ctx, state, plan, lambda: time[0]), ctx, time)
    assert state.engaged and state.returning
    assert any(s.kind == 'key' and s.keys == ('s',) for s in actions)
    assert not any(s.kind == 'click' and s.button == 'left' for s in actions)
    time[0] += plan['check_every']
    assert miners.due(army, time[0])
    monkeypatch.setattr(miners, 'ore_target', lambda *args: (620, 620))
    actions = drive(miners.support(ctx, state, plan, lambda: time[0]), ctx, time)
    assert not state.engaged and any(s.point == (620, 620) for s in actions)


def test_interrupted_attack_keeps_return_handling_active(monkeypatch):
    ctx, state, plan, time, army = support_setup(monkeypatch)
    task = miners.support(ctx, state, plan, lambda: time[0])
    value = None
    for _ in range(40):
        step = task.send(value)
        value = step.fn(ctx.world.snapshot(), None) if isinstance(step, Look) else None
        if isinstance(step, Act) and any(s.keys == ('ctrl', 'shift') for s in step.steps):
            break
    else:
        raise AssertionError('No defense order')
    task.close()
    assert state.engaged
    time[0] += plan['check_every']
    assert miners.due(army, time[0])


@pytest.mark.parametrize('placing', [False, True])
def test_return_deadline_preempts_defense_but_respects_active_placement(placing):
    from addons.chronodivide.tests.test_strategy import setup
    world, ctx, runner, army, reflex, executor, time = setup()
    plan = army.book['defense']['emergency_miners']
    army.miners.engaged = True
    army.miners.started = time[0] - plan['max_seconds']

    def occupied():
        yield Poll(lambda s, v: False, 100)

    runner.start('army:defend', 'army', occupied(), priority=2)
    if placing:
        runner.start('place:barracks', 'sidebar', occupied())
    army.step(world.snapshot())
    assert runner.tasks['army'].name == ('army:defend' if placing else 'army:miner-support')
    if not placing:
        assert runner.finished[-1].status == 'cancelled'


def test_ore_pixels_survive_native_and_independent_browser_frames():
    native = cv2.imread(str(FIXTURES / 'frame-play.png'))
    point = ore_target(native, (425, 368), 600)
    assert point and 115 <= point[0] <= 305 and 330 <= point[1] <= 437
    browser = cv2.imread(str(FIXTURES / 'ore-field-browser.png'))
    point = ore_target(browser, (0, 0), 700, (0, 0, browser.shape[1], browser.shape[0]))
    assert point is not None
    b, g, r = map(int, browser[point[1], point[0]])
    assert r >= 235 and g >= 210 and r - b >= 90


@pytest.mark.parametrize('fixture', ['base-reactors.png', 'frame-start.png', 'reactor-glow-browser.png'])
def test_no_ore_target_on_buildings_shroud_or_empty_ground(fixture):
    image = cv2.imread(str(FIXTURES / fixture))
    region = (0, 0, min(850, image.shape[1]), min(736, image.shape[0]))
    assert ore_target(image, (425, 368), 1000, region) is None


def test_ore_reader_does_not_use_sidebar_or_isolated_gold_specks():
    image = np.zeros((768, 1024, 3), np.uint8)
    for y in range(300, 350, 5):
        image[y:y + 2, 900:903] = (50, 240, 250)
    image[400:402, 400:403] = (50, 240, 250)
    assert ore_target(image, (425, 368), 1000) is None
