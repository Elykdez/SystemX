"""Unattended supervision: resume after a quiet period, pause-menu recovery of pointer capture, user hold."""

from types import SimpleNamespace
import threading

from desktop.input import Gate
from addons.chronodivide.player import Player
from addons.chronodivide.unattended import Supervisor


class FakePlayer:
    def __init__(self):
        self.physical, self.physical_at, self.free = False, None, False
        self.menu_since, self.menu_by_user = None, False
        self.last_playing = (False, 'game does not hold the pointer')
        self.calls = []

    def taken_over(self):
        return self.physical

    def last_physical(self):
        return self.physical_at

    def wants_pointer(self):
        return self.free

    def menu_open(self):
        return self.menu_since is not None

    def arm_on_start(self):
        self.calls.append('arm')
        self.physical = False

    def pause(self, reason=''):
        self.calls.append('pause')

    def recapture_pointer(self):
        self.calls.append('click')

    def resume_menu(self):
        self.calls.append('resume')
        self.menu_since = None


def make():
    now = [100.0]
    player = FakePlayer()
    return player, Supervisor(player, resume_after=8.0, grab_after=2.0, clock=lambda: now[0]), now


def test_resumes_once_after_the_quiet_period():
    player, supervisor, now = make()
    player.physical, player.physical_at = True, 99.0
    assert supervisor.step() is None                    # the user is still moving the mouse
    now[0] = 106.9
    assert supervisor.step() is None
    now[0] = 107.0
    assert supervisor.step() == 'resumed'
    assert supervisor.step() is None and player.calls == ['arm']


def test_free_pointer_is_recaptured_by_a_map_click_with_back_off():
    player, supervisor, now = make()
    player.free = True
    assert supervisor.step() is None                    # just noticed
    now[0] += 2.0
    assert supervisor.step() == 'recaptured' and player.calls == ['click']
    now[0] += 2.0
    assert supervisor.step() is None                    # second try waits twice as long
    now[0] += 2.0
    assert supervisor.step() == 'recaptured'
    player.free, player.last_playing = False, (True, 'playing')
    assert supervisor.step() is None and supervisor.attempts == 0     # captured: counter reset


def test_gives_up_recapturing_after_a_few_tries():
    player, supervisor, now = make()
    player.free = True
    for _ in range(40):
        supervisor.step()
        now[0] += 1.0
    assert player.calls == ['click'] * 5


def test_the_pause_menu_is_resumed_when_the_user_did_not_open_it():
    player, supervisor, now = make()
    player.menu_since = now[0]
    now[0] += 1.0
    assert supervisor.step() is None                    # menu just appeared
    now[0] += 0.5
    assert supervisor.step() == 'resumed-menu' and player.calls == ['resume']


def test_a_menu_the_user_opened_waits_for_a_long_quiet_time_or_f8():
    player, supervisor, now = make()
    player.menu_since, player.menu_by_user, player.physical_at = 100.0, True, 99.5
    now[0] = 130.0
    assert supervisor.step() is None                    # the user paused the game on purpose
    now[0] = 160.0
    assert supervisor.step() == 'resumed-menu'
    player.menu_since, player.menu_by_user, player.physical_at = 200.0, True, 199.5
    now[0] = 202.0
    supervisor.hold()
    supervisor.release()                                # F10 then F8: resume now
    assert supervisor.step() == 'resumed-menu'


def test_user_hold_blocks_resume_and_recovery_until_released():
    player, supervisor, now = make()
    supervisor.hold()
    player.physical, player.free, player.menu_since = True, True, 90.0
    now[0] = 500.0
    assert supervisor.step() is None and player.calls == ['pause']
    supervisor.release()
    assert player.calls == ['pause', 'arm'] and not supervisor.held


def test_front_run_restores_window_after_quiet_and_waits_for_fresh_state():
    player, supervisor, now = make()
    foreground = [False]
    player.in_front = lambda: foreground[0]
    def restore():
        player.calls.append('front')
        foreground[0] = True
        return True
    supervisor.front = restore
    player.physical, player.physical_at = True, 99.0
    player.menu_since = 90.0
    assert supervisor.step() is None and not player.calls
    now[0] = 107.0
    assert supervisor.step() == 'restored-window' and player.calls == ['front']
    assert supervisor.step() == 'resumed'
    assert supervisor.step() == 'resumed-menu'


def test_failed_window_restore_backs_off_and_user_hold_prevents_it():
    player, supervisor, now = make()
    player.in_front = lambda: False
    supervisor.front = lambda: player.calls.append('front')
    assert supervisor.step() is None and player.calls == ['front']
    now[0] += 1
    assert supervisor.step() is None and player.calls == ['front']
    now[0] += 7
    assert supervisor.step() is None and player.calls == ['front', 'front']
    supervisor.hold()
    now[0] += 100
    assert supervisor.step() is None and player.calls == ['front', 'front', 'pause']


def test_arm_on_start_ends_a_takeover_pause_while_waiting_for_play():
    player = object.__new__(Player)
    player.mode, player.last_playing = 'execute', (False, 'game does not hold the pointer')
    player._physical, player.pending_arm = True, False
    player.events = lambda entry: None
    player.arm_on_start()
    assert player.pending_arm and not player.taken_over()


def test_agent_pause_keeps_the_reason_and_explains_that_the_game_can_continue():
    player = object.__new__(Player)
    player.gate, player.status, player.pending_arm = Gate(), {}, True
    player.gate.start()
    cancelled = []
    player.runner = SimpleNamespace(cancel_all=cancelled.append)
    player.pointer = None
    player.events = lambda entry: None
    player.pause('pointer calibration failed')
    assert not player.gate.running and not player.pending_arm
    assert cancelled == ['pointer calibration failed']
    assert player.pause_reason == 'pointer calibration failed'
    assert player.status['state'] == 'paused'
    assert 'game may still be running' in player.status['reason']


def test_capture_failure_invalidates_play_and_menu_recovery(monkeypatch):
    from desktop.target import DesktopPause
    player = object.__new__(Player)
    player.stop_event = threading.Event()
    player.last_playing, player.pointer_free, player.menu_since = (True, 'playing'), True, 100
    player.status, player.tick_period = {}, .05
    def fail_capture():
        player.stop_event.set()
        raise DesktopPause('Target must be visible and not minimized.')
    player.tick, player.close = fail_capture, lambda: None
    monkeypatch.setattr('addons.chronodivide.player.time.sleep', lambda _: None)
    player.run()
    assert player.last_playing == (False, 'Target must be visible and not minimized.')
    assert not player.pointer_free and player.menu_since is None
    assert player.status['state'] == 'waiting'
