from dataclasses import replace

import cv2
import pytest

from addons.chronodivide.army import LOST_AFTER, battle_session, clear_factory_exit
from addons.chronodivide.world import Slot
from addons.chronodivide.tests.test_strategy import (
    CameraGame, FIXTURES, attack_setup, online_radar, run, setup,
)
from addons.chronodivide.tests.test_unitboxes import draw_bar


@pytest.mark.parametrize('item,stale,threat,clears', [
    ('rhino_tank', False, False, True),
    ('war_miner', False, False, True),
    ('harrier', False, False, False),
    ('rhino_tank', True, False, False),
    ('rhino_tank', False, True, False),
])
def test_finished_ground_unit_requires_persistent_fresh_quiet_evidence_to_clear_exit(
        item, stale, threat, clears):
    world, ctx, runner, army, _, _, time = setup()
    ctx.camera = 'home'
    world.units_built['rhino_tank'] = 4
    world.slots['vehicles'].value = [Slot(0, item, 'ready')]
    world.slots['vehicles'].at = time[0]
    first = army.step(world.snapshot())
    assert first is None or first[0] != 'army:clear-exit'
    time[0] += 3.1
    if not stale:
        world.slots['vehicles'].at = time[0]
    if threat:
        army.threat_at = time[0]
    plan = army.step(world.snapshot())
    assert (plan is not None and plan[0] == 'army:clear-exit') == clears
    if clears:
        runner.cancel_all('test completed')
        time[0] += 3.1
        world.slots['vehicles'].at = time[0]
        assert army.step(world.snapshot()) is None  # bounded retries


def test_clearing_factory_exit_requires_verified_vehicles():
    import numpy as np
    world, ctx, runner, army, _, executor, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    runner.start('army:clear-exit', 'army', clear_factory_exit(ctx))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert not any(e['kind'] == 'order' for e in world.events)
    ctx.last_canvas[:] = (50, 140, 90)
    draw_bar(ctx.last_canvas, 400, 200, 36)
    runner.start('army:clear-exit', 'army', clear_factory_exit(ctx))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert any(e['kind'] == 'order' and e['order'] == 'clear_factory_exit' for e in world.events)
    assert any(step[2] == ('ctrl', '1') for act in executor.acts for step in act[2])
    point = next(e['at'] for e in world.events if e.get('order') == 'clear_factory_exit')
    assert abs(point[0] - 418) + abs(point[1] - 225) >= 80
    assert any(step[0] == 'click' and step[1] == tuple(point)
               and step[2] == ('ctrl', 'shift') for act in executor.acts for step in act[2])


def test_factory_exit_recovery_does_not_order_onto_unverified_ground():
    import numpy as np
    world, ctx, runner, army, _, _, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    draw_bar(ctx.last_canvas, 400, 200, 36)
    runner.start('army:clear-exit', 'army', clear_factory_exit(ctx))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert not any(e.get('order') == 'clear_factory_exit' for e in world.events)


def test_factory_exit_check_survives_a_slow_sidebar_round_trip():
    world, ctx, runner, army, _, _, time = setup()
    ctx.camera = 'home'
    world.units_built['rhino_tank'] = 4
    world.slots['vehicles'].value = [Slot(0, 'war_miner', 'ready')]
    world.slots['vehicles'].at = time[0]
    army.step(world.snapshot())
    runner.cancel_all('sidebar observation')
    time[0] += 45
    world.slots['vehicles'].at = time[0]
    assert army.step(world.snapshot())[0] == 'army:clear-exit'


@pytest.mark.parametrize('formed, enabled', [(False, True), (True, False)])
def test_unavailable_team_cannot_confirm_loss_or_stall_the_next_wave(formed, enabled):
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=12)
    army.step(world.snapshot())
    runner.cancel_all('interrupted wave')
    army.campaign.team_formed, army.campaign.teams = formed, enabled
    army.campaign.loss_confirmed = True
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'base-reactors.png'))
    runner.start('army:verify', 'army', army._verify_army())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert not army.campaign.loss_confirmed and executor.acts == []
    time[0] += 90
    assert army._attack(time[0], True)[0] == 'army:attack'
    assert army.campaign.active and army.failures == 0
    assert runner.tasks['army'].name == 'army:attack'


def test_interrupted_verification_marks_camera_lost_and_recovers_home():
    world, ctx, runner, army, reflex, executor, time = setup()
    army.campaign.team_formed = True
    runner.start('army:verify', 'army', army._verify_army())
    runner.tick(world.snapshot(), None)
    runner.cancel_all('paused before home acknowledgement')
    assert ctx.camera == 'lost'
    assert army.step(world.snapshot()) == ('army:camera', None)
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert ctx.camera == 'home'


@pytest.mark.parametrize('failures, extra, depleted', [(0, 0, False), (1, 0, True), (1, 1, True), (1, 2, True), (1, 3, False)])
def test_recorded_survivors_regroup_only_after_an_earlier_failed_attack(failures, extra, depleted):
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=12)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    army.failures = failures
    runner.executor = executor = CameraGame(world)
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'attack-remnants-browser.png'))
    for index in range(extra):
        draw_bar(ctx.last_canvas, 500 + 50 * index, 180, 36)
    ctx.compressed_frames = True
    runner.start('army:verify', 'army', army._verify_army())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    note = next(e for e in world.events if e.get('kind') == 'army_verified')
    assert note['units'] == [2 + extra, 2 + extra] and note['vehicles'] == [2 + extra, 2 + extra]
    assert army.campaign.depleted_confirmed == depleted
    assert not army.campaign.loss_confirmed
    if depleted:
        assert army._attack(time[0], True) is None
        assert not army.campaign.active and army.failures == failures + 1
        assert army.intent.posture == 'gather' and army.intent.attack_size is None


def test_cached_front_with_zero_visible_selected_units_cannot_issue_battle_orders():
    world, ctx, runner, army, reflex, executor, time = setup()
    runner.executor = executor = CameraGame(world)
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-near-base.png'))
    army.campaign.confirmed_front = (942, 122)
    army.campaign.army_seen_at = time[0]
    army.campaign.front_seen_at = time[0]
    runner.start('army:focus', 'army', battle_session(ctx, army.campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    note = next(e for e in world.events if e.get('order') == 'focus')
    assert note['units_seen'] == [0] and note['orders'] == []
    clicks = [s for a in executor.acts for s in a[2] if s[0] == 'click' and s[3] == 'left']
    assert [s[1] for s in clicks] == [(942, 122)]  # the only left click moved the camera
    assert ctx.camera == 'home'


def test_cached_front_expires_before_a_new_camera_or_battle_order():
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    army.campaign.confirmed_front = (942, 122)
    army.campaign.army_seen_at = time[0] - LOST_AFTER - 0.1
    army.campaign.front_seen_at = time[0] - LOST_AFTER - 0.1
    runner.start('army:focus', 'army', battle_session(ctx, army.campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert executor.acts == [] and ctx.camera == 'home'


@pytest.mark.parametrize('visible', [False, True])
def test_active_team_without_field_dots_is_observed_even_when_camera_does_not_move(visible):
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-near-base.png'))
    if visible:
        draw_bar(ctx.last_canvas, 407, 340, 36)
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.focus_misses = 1
    started = time[0]
    runner.start('army:focus', 'army', battle_session(ctx, campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    taps = [step[2] for step in executor.acts[0][2] if step[0] == 'key']
    assert taps == [('1',), ('1',)]
    note = next(e for e in world.events if e.get('order') == 'focus')
    assert not note['camera'] and note['units_seen'] == [int(visible)]
    clicks = [step for act in executor.acts for step in act[2] if step[0] == 'click' and step[3] == 'left']
    assert len(clicks) == int(visible)  # one bounded order, only with freshly selected units
    if visible:
        assert clicks[0][2] == ('ctrl', 'shift')
        assert campaign.teams and campaign.focus_misses == 0
        assert campaign.army_seen_at >= started and campaign.confirmed_front == radar.view_center
    else:
        assert campaign.army_seen_at is None and campaign.confirmed_front is None
    assert ctx.camera == 'home' and not campaign.loss_confirmed


@pytest.mark.parametrize('survivors', [False, True])
def test_empty_prism_team_checks_the_whole_army_before_counting_a_focus_miss(monkeypatch, survivors):
    import numpy as np
    from addons.chronodivide import allied_tactics, playbook
    from addons.chronodivide.tests.test_strategy import Recorder

    world, ctx, runner, army, _, _, time = setup()
    army.book = playbook.load('addons/chronodivide/profiles/startlux-ra2-allied.json')
    world.units_built['prism_tank'] = 4
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.prism_roles = {'prism_tank': '5', 'ifv': '7'}
    campaign.focus_misses = 1
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)

    class SelectionGame(Recorder):
        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            if any(step.kind == 'key' and step.keys == ('1',) for step in steps) and survivors:
                draw_bar(ctx.last_canvas, 407, 340, 36)

    runner.executor = executor = SelectionGame()
    def quiet_round(*args):
        yield from ()
        return []
    monkeypatch.setattr(allied_tactics, 'round_order', quiet_round)
    runner.start('army:focus', 'army', battle_session(ctx, campaign, army=army))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    keys = [step[2] for act in executor.acts for step in act[2] if step[0] == 'key']
    assert keys[:5] == [('5',), ('5',), ('5',), ('1',), ('1',)]
    note = next(e for e in world.events if e.get('order') == 'focus')
    assert note['units_seen'] == [int(survivors)]
    assert campaign.teams == survivors
    assert campaign.focus_misses == (0 if survivors else 2)
    assert 'prism_tank' not in campaign.prism_roles
    assert campaign.prism_roles['ifv'] == '7'


def test_cleared_prism_selection_is_rechecked_before_discarding_its_saved_role(monkeypatch):
    import numpy as np
    from addons.chronodivide import allied_tactics, playbook
    from addons.chronodivide.tests.test_strategy import Recorder
    world, ctx, runner, army, _, _, time = setup()
    army.book = playbook.load('addons/chronodivide/profiles/startlux-ra2-allied.json')
    world.units_built['prism_tank'] = 4
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.prism_roles = {'prism_tank': '5', 'ifv': '7'}
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)

    class ClearedSelection(Recorder):
        taps = 0
        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            self.taps += sum(s.kind == 'key' and s.keys == ('5',) for s in steps)
            if self.taps >= 3:
                draw_bar(ctx.last_canvas, 407, 340, 36)

    def quiet_round(*args):
        yield from ()
        return []
    monkeypatch.setattr(allied_tactics, 'round_order', quiet_round)
    runner.executor = executor = ClearedSelection()
    runner.start('army:focus', 'army', battle_session(ctx, campaign, army=army))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=400)
    assert campaign.prism_roles['prism_tank'] == '5' and campaign.teams
    assert campaign.focus_misses == 0
    assert not any(s[0] == 'key' and s[2] == ('1',) for a in executor.acts for s in a[2])


def test_fresh_general_army_contact_does_not_refresh_an_old_camera_position():
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.confirmed_front = (942, 122)
    campaign.front_seen_at = time[0] - LOST_AFTER - .1
    campaign.army_seen_at = time[0]
    runner.start('army:focus', 'army', battle_session(ctx, campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    taps = [s[2] for a in executor.acts for s in a[2] if s[0] == 'key']
    assert taps[:2] == [('1',), ('1',)]
    clicks = [s[1] for a in executor.acts for s in a[2] if s[0] == 'click' and s[3] == 'left']
    assert (942, 122) not in clicks


def test_repeated_siege_orders_cannot_postpone_verification_of_a_missing_field_army():
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.target = (942, 122)
    campaign.army_seen_at = time[0] - LOST_AFTER - 1
    campaign.last_wave = time[0] - 1
    assert army._army_lost(time[0], world.radar.value)


def test_loss_verification_checks_the_battle_location_instead_of_an_empty_team_centroid():
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.target = radar.enemy[0][1]
    army.failures = 1
    empty = cv2.imread(str(FIXTURES / 'base-reactors.png'))
    survivors = cv2.imread(str(FIXTURES / 'attack-remnants-browser.png'))
    for index in range(3):
        draw_bar(survivors, 500 + 50 * index, 180, 36)
    ctx.last_canvas, ctx.compressed_frames = empty, True

    class SeparatedTeam(CameraGame):
        group_keys = 0

        def act(self, name, lane, steps, commands=1):
            result = super().act(name, lane, steps, commands)
            for step in steps:
                if step.kind == 'click' and step.point == campaign.target and step.button == 'left':
                    ctx.last_canvas = survivors
                elif step.kind == 'key' and step.keys == ('1',):
                    self.group_keys += 1
                    if self.group_keys >= 2:
                        ctx.last_canvas = empty
            return result

    runner.executor = executor = SeparatedTeam(world)
    runner.start('army:verify', 'army', army._verify_army())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    note = next(e for e in world.events if e.get('kind') == 'army_verified')
    assert note['units'] == [5, 5]
    assert not campaign.loss_confirmed and not campaign.depleted_confirmed
    taps = [s[2] for a in executor.acts for s in a[2] if s[0] == 'key']
    assert taps.count(('1',)) == 1


def test_unavailable_battle_view_cannot_confirm_an_army_loss():
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    world.set('radar', radar)
    ctx.radar_home = radar.own[0][1]
    army.campaign.active = army.campaign.team_formed = True
    army.campaign.target = radar.enemy[0][1]
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'base-reactors.png'))
    runner.executor = CameraGame(world, follows=False)
    runner.start('army:verify', 'army', army._verify_army())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert not army.campaign.loss_confirmed and not army.campaign.depleted_confirmed
    assert ctx.camera == 'home'
    note = next(e for e in world.events if e.get('kind') == 'army_verified')
    assert note['reason'] == 'battle view unavailable'


def test_active_attack_inspects_its_target_when_field_dots_are_missing():
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.target = radar.enemy[0][1]
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'enemy-near-base.png'))
    draw_bar(ctx.last_canvas, 407, 340, 36)
    runner.executor = executor = CameraGame(world)
    runner.start('army:focus', 'army', battle_session(ctx, campaign, seconds=1, every=1))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    note = next(e for e in world.events if e.get('order') == 'focus')
    assert note['front'] == list(campaign.target) and note['units_seen'] == [1]
    assert note['orders'] == ['fight']


@pytest.mark.parametrize('active, formed, enabled', [(False, True, True), (True, False, True), (True, True, False)])
def test_no_field_dots_without_an_established_active_team_remains_a_noop(active, formed, enabled):
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    army.campaign.active, army.campaign.team_formed, army.campaign.teams = active, formed, enabled
    runner.start('army:focus', 'army', battle_session(ctx, army.campaign))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert executor.acts == [] and ctx.camera == 'home'


def test_repeated_empty_online_focus_preserves_team_for_normal_loss_verification():
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=12)
    army.step(world.snapshot())
    assert run(runner, world, time, lambda: not runner.busy('army'))
    radar = online_radar()
    home = radar.own[0][1]
    world.set('radar', replace(radar, own=((48, home),)))
    ctx.radar_home = home
    ctx.last_canvas = cv2.imread(str(FIXTURES / 'base-reactors.png'))
    runner.executor = CameraGame(world)
    campaign = army.campaign
    campaign.army_seen_at = campaign.last_wave = time[0] - LOST_AFTER - 1
    for _ in range(3):
        runner.start('army:focus', 'army', battle_session(ctx, campaign))
        assert run(runner, world, time, lambda: not runner.busy('army'))
        assert campaign.teams and campaign.active and not campaign.loss_confirmed
    assert army._attack(time[0], True)[0] == 'army:verify'
    assert runner.tasks['army'].name == 'army:verify'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert campaign.loss_confirmed
    army._attack(time[0], True)
    assert not army.campaign.active and army.failures == 1
