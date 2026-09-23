import sys
from types import SimpleNamespace

import pytest

if sys.platform != 'win32':
    pytest.skip('Windows input layer', allow_module_level=True)

from desktop.input import Gate
from desktop.target import DesktopPause, Geometry
from addons.chronodivide.executor import Executor, Guard, LockedPointer, Step, AbsolutePointer
from addons.chronodivide.window import Rect

GEOMETRY = Geometry(100, 50, 1200, 900, 96)


class FakeTarget:
    hwnd = 42

    def __init__(self):
        self.points = []

    def geometry(self):
        return GEOMETRY

    def check_point(self, x, y, foreground=False):
        self.points.append((x, y))


def setup(foreground=42, takeover=lambda: False, sleep=lambda s: None):
    gate = Gate()
    gate.start()
    view = SimpleNamespace(geometry=GEOMETRY, canvas=Rect(88, 66, 1024, 768))
    target = FakeTarget()
    guard = Guard(target, view, gate, gate.generation, takeover=takeover, check_ime=False,
                  foreground=lambda: foreground)
    sent = []
    executor = Executor(guard, AbsolutePointer(guard), sender=lambda ev: sent.append(ev) or len(ev),
                        sleep=sleep, step_gap_ms=0)
    return executor, guard, gate, target, sent


def test_click_maps_canvas_point_to_desktop_and_sends_one_batch_per_step():
    executor, guard, gate, target, sent = setup()
    executor.act('t', 'lane', [Step.click((500, 400)), Step.key('q')])
    assert target.points == [(100 + 88 + 500, 50 + 66 + 400)]
    assert [len(batch) for batch in sent] == [1, 2, 2]  # move, (rest) down+up, key down+up
    assert executor.counts['transactions'] == 1


def test_repair_hotkey_is_narrowly_authorized_and_still_honors_blocked_keys():
    executor, guard, *_ = setup()
    with pytest.raises(DesktopPause):
        executor.act('repair:yard', 'sidebar', [Step.key('k')])
    with pytest.raises(DesktopPause):
        executor.act('queue:reactor', 'sidebar', [Step.repair_mode()])
    executor.act('repair:yard', 'sidebar', [Step.repair_mode()])
    guard.blocked.add(('k',))
    with pytest.raises(DesktopPause):
        executor.act('repair:yard', 'sidebar', [Step.repair_mode()])


def test_modifier_click_holds_the_modifiers_around_the_click_in_stages():
    executor, *_, sent = setup()
    executor.act('t', 'lane', [Step.click((500, 400), modifiers=('ctrl', 'shift'))])
    # keys down, then the move, then the button, then keys up: separate batches with pauses between,
    # so the game sees the modifiers held on the frame it handles the click
    assert [len(batch) for batch in sent] == [2, 1, 2, 2]
    assert all(e.type == 1 for e in sent[0] + sent[3]) and all(e.type == 0 for e in sent[1] + sent[2])


@pytest.mark.parametrize('attack_move', [False, True])
def test_waypoint_route_holds_z_across_all_clicks_then_releases_it(attack_move):
    from dataclasses import asdict
    from desktop import input as di
    executor, guard, gate, target, sent = setup()
    route = Step.route([(300, 250), (550, 400)], (362, 754), attack_move=attack_move)
    assert asdict(route)['points'] == ((300, 250), (550, 400))
    executor.act('route', 'army', [route])
    held, clicks = set(), []
    for batch in sent:
        for event in batch:
            if event.type == 1:
                key = event.data.ki.wVk
                if event.data.ki.dwFlags & 2:
                    held.discard(key)
                else:
                    held.add(key)
            elif event.data.mi.dwFlags == 2:
                clicks.append(held.copy())
    expected = {di.VK['z']} | ({di.VK['ctrl'], di.VK['shift']} if attack_move else set())
    assert clicks == [expected, expected] and not held
    assert executor.counts['transactions'] == 1 and executor.counts['commands'] == 1
    assert (100 + 88 + 362, 50 + 66 + 754) not in target.points  # native never clicks the button


@pytest.mark.parametrize('points', [[], [(300, 250)] * 17])
def test_waypoint_route_requires_bounded_nonempty_points(points):
    with pytest.raises(ValueError, match='1 to 16'):
        Step.route(points, (362, 754))


def test_waypoint_route_checks_all_destinations_and_ime_before_input():
    executor, guard, gate, target, sent = setup()
    with pytest.raises(DesktopPause, match='safe canvas zone'):
        executor.act('route', 'army', [Step.route([(300, 250), (5, 400)], (362, 754))])
    assert sent == []
    guard.check_ime = True
    guard.ime_native = lambda: True
    with pytest.raises(DesktopPause, match='IME'):
        executor.act('route', 'army', [Step.route([(300, 250)], (362, 754))])
    assert sent == []


def test_waypoint_route_focus_loss_releases_z_without_more_clicks(monkeypatch):
    from desktop import input as di
    executor, guard, gate, target, sent = setup()
    monkeypatch.setattr(di, 'release_all', lambda **kwargs: None)
    # Lose foreground after the first queued click. Cleanup must contain only releases,
    # never a Z tap or another click aimed at whichever application now has focus.
    def sleep(seconds):
        if any(e.type == 0 and e.data.mi.dwFlags == 2 for batch in sent for e in batch):
            guard.foreground = lambda: 7
    executor.sleep = sleep
    with pytest.raises(DesktopPause, match='not in front'):
        executor.act('route', 'army', [Step.route([(300, 250), (550, 400)], (362, 754))])
    assert sum(e.type == 0 and e.data.mi.dwFlags == 2 for batch in sent for e in batch) == 1
    assert len(sent[-1]) == 1
    assert sent[-1][0].data.ki.wVk == di.VK['z'] and sent[-1][0].data.ki.dwFlags & 2


def test_waypoint_route_pause_before_first_click_releases_z(monkeypatch):
    from desktop import input as di
    executor, guard, gate, target, sent = setup()
    monkeypatch.setattr(di, 'release_all', lambda **kwargs: None)
    def sleep(seconds):
        if len(sent) >= 2:             # initial Z tap then held Z
            gate.pause()
    executor.sleep = sleep
    with pytest.raises(DesktopPause, match='cancelled'):
        executor.act('route', 'army', [Step.route([(300, 250)], (362, 754))])
    assert all(e.type == 1 for batch in sent for e in batch)
    assert sent[-1][0].data.ki.wVk == di.VK['z'] and sent[-1][0].data.ki.dwFlags & 2


@pytest.mark.parametrize('point', [(5, 400), (500, 760), (1020, 10)])
def test_clicks_near_canvas_edges_are_refused(point):
    executor, *_, sent = setup()
    with pytest.raises(DesktopPause, match='safe canvas zone'):
        executor.act('t', 'lane', [Step.click(point)])
    assert sent == []


@pytest.mark.parametrize('keys', [('f5',), ('ctrl', 'w'), ('alt', 'f4'), ('backspace',)])
def test_browser_shortcuts_are_blocked(keys):
    executor, *_, sent = setup()
    with pytest.raises(DesktopPause, match='blocked'):
        executor.act('t', 'lane', [Step.chord(*keys)])
    assert sent == []


def test_focus_loss_takeover_and_pause_prevent_input():
    for kwargs, message in [(dict(foreground=7), 'not in front'), (dict(takeover=lambda: True), 'paused')]:
        executor, *_, sent = setup(**kwargs)
        with pytest.raises(DesktopPause, match=message):
            executor.act('t', 'lane', [Step.key('q')])
        assert sent == []
    executor, guard, gate, target, sent = setup()
    gate.pause()
    with pytest.raises(DesktopPause, match='cancelled'):
        executor.act('t', 'lane', [Step.key('q')])


def test_pause_between_steps_stops_the_rest_of_a_transaction():
    executor, guard, gate, target, sent = setup()
    executor.sleep = lambda s: gate.pause()
    with pytest.raises(DesktopPause):
        executor.act('t', 'lane', [Step.key('q'), Step.wait(10), Step.key('s')])
    assert len(sent) == 1


def test_locked_pointer_needs_a_known_cursor_then_moves_by_exact_deltas():
    executor, guard, *_ = setup()
    pointer = LockedPointer(guard, cursor=lambda: (700, 500), clip=lambda: (0, 0, 2000, 2000))
    from addons.chronodivide.tasks import TaskFailed
    with pytest.raises(TaskFailed, match='unknown'):              # the task fails; no pause
        pointer.move_events((300, 200))
    pointer.observe((300, 200))
    events = pointer.move_events((350, 150))
    assert len(events) == 1 and pointer.position == (350, 150)


class ChromeLock:
    """The OS cursor under Chrome's pointer lock: clipped to the window, and pulled back to the
    window's center once it enters the outer 15% band of it."""

    def __init__(self, window=(100, 50, 1700, 1250), at=(900, 300)):
        self.window, self.at, self.moves = window, at, []

    def cursor(self):
        return self.at

    def clip(self):
        return self.window

    def send(self, events):
        left, top, right, bottom = self.window
        before = self.at
        self.at = self.target                       # set by jump through the patched mouse_events
        self.moves.append((self.at[0] - before[0], self.at[1] - before[1]))
        bx, by = (right - left) * 0.15, (bottom - top) * 0.15
        x, y = self.at
        if x < left + bx or x > right - bx or y < top + by or y > bottom - by:
            self.at = ((left + right) // 2, (top + bottom) // 2)
        return len(events)


def test_long_moves_go_in_steps_so_the_window_edge_never_cuts_them_short(monkeypatch):
    executor, guard, *_ = setup()
    lock = ChromeLock()
    from desktop import input as di
    original = di.mouse_events

    def mouse_events(point, desktop, button=None, move=True):
        if point is not None:
            lock.target = (round(point[0]), round(point[1]))
        return original(point, desktop, button, move)
    monkeypatch.setattr(di, 'mouse_events', mouse_events)
    pointer = LockedPointer(guard, cursor=lock.cursor, clip=lock.clip)
    pointer.observe((940, 690))
    pointer.move((909, 94), lock.send, lambda s: None)          # 596 px up: the radar from the park
    assert pointer.position == (909, 94)
    assert sum(dy for _, dy in lock.moves) == 94 - 690          # all of it reached the game
    assert max(max(abs(dx), abs(dy)) for dx, dy in lock.moves) <= 120


def test_calibration_nudge_delivers_all_motion_near_the_chrome_window_edge(monkeypatch):
    executor, guard, *_ = setup()
    lock = ChromeLock(window=(0, 0, 1074, 892), at=(537, 716))
    from desktop import input as di

    def mouse_events(point, desktop, button=None, move=True):
        lock.target = tuple(point)
        return [object()]

    monkeypatch.setattr(di, 'mouse_events', mouse_events)
    executor.pointer = LockedPointer(guard, cursor=lock.cursor, clip=lock.clip)
    executor.sender = lock.send
    # The previous single jump delivered only 174 px, reporting 260 / 174 = 1.49 gain.
    assert executor.pointer.jump(0, 260)[1] == (0, 174)
    executor.act('calibrate_pointer', 'sidebar', [Step.nudge(0, 260)], commands=0)
    assert sum(dy for _, dy in lock.moves) == 260
    assert executor.pointer.position is None      # vision still establishes the game position


def test_relative_locked_motion_does_not_stall_at_an_os_cursor_clip_edge():
    executor, guard, *_ = setup()
    pointer = LockedPointer(guard, relative=True, cursor=lambda: (1072, 1), clip=lambda: (0, 0, 1074, 892))
    sent = []

    def send(events):
        sent.extend((event.data.mi.dx, event.data.mi.dy, event.data.mi.dwFlags) for event in events)

    pointer.nudge(1100, -1100, send, lambda _: None)
    assert sum(dx for dx, _, _ in sent) == 1100 and sum(dy for _, dy, _ in sent) == -1100
    assert all(flags == 1 for _, _, flags in sent)
    assert all(max(abs(dx), abs(dy)) <= 120 for dx, dy, _ in sent)


@pytest.mark.parametrize('relative', [False, True])
def test_full_calibration_recovers_the_pointer_in_a_small_chrome_window(monkeypatch, relative):
    from addons.chronodivide.executor import Trust
    from addons.chronodivide.layout import Layout
    from addons.chronodivide.skills import Context, calibrate_pointer
    from addons.chronodivide.tasks import TaskRunner
    from addons.chronodivide.world import World
    from desktop import input as di
    import numpy as np

    executor, guard, *_ = setup()
    lock = ChromeLock(window=(0, 0, 1074, 892), at=(537, 446))
    game_cursor = [425, 360]
    if relative:
        lock.at = (1072, 1)                   # the OS cursor need not recenter for relative input

    def mouse_events(point, desktop, button=None, move=True):
        lock.target = tuple(point)
        return [object()]

    def send(events):
        if relative:
            delta = tuple(sum(getattr(event.data.mi, axis) for event in events) for axis in ('dx', 'dy'))
            count = len(events)
        else:
            count = lock.send(events)
            delta = lock.moves[-1]
        for axis, limit in enumerate((1023, 767)):
            game_cursor[axis] = max(0, min(limit, game_cursor[axis] + delta[axis]))
        return count

    monkeypatch.setattr(di, 'mouse_events', mouse_events)
    executor.pointer = LockedPointer(guard, cursor=lock.cursor, clip=lock.clip, relative=relative)
    executor.sender = send
    clock = [100.0]
    world = World(clock=lambda: clock[0])
    finder = SimpleNamespace(find=lambda canvas, region=None: tuple(game_cursor))
    ctx = Context(Layout.for_canvas(1024, 768), world, finder, executor.pointer)
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    runner = TaskRunner(executor, Trust(), clock=lambda: clock[0])
    runner.start('calibrate_pointer', 'sidebar', calibrate_pointer(ctx))
    for _ in range(20):
        runner.tick(world.snapshot(), None)
        clock[0] += 0.2
        if not runner.busy('sidebar'):
            break
    assert runner.finished[-1].status == 'done', runner.finished[-1].reason
    assert ctx.gain_measured and executor.pointer.gain == 1.0
    assert executor.pointer.position == tuple(game_cursor) == ctx.park


def test_pausing_during_a_calibration_nudge_stops_remaining_motion(monkeypatch):
    executor, guard, gate, _, sent = setup()
    executor.pointer = LockedPointer(guard)
    monkeypatch.setattr(executor.pointer, 'jump', lambda dx, dy: ([object()], (dx, dy)))
    executor.sleep = lambda seconds: gate.pause()
    with pytest.raises(DesktopPause, match='cancelled'):
        executor.act('calibrate_pointer', 'sidebar', [Step.nudge(0, 260)], commands=0)
    assert len(sent) == 1


def test_paused_game_blocks_every_step():
    executor, guard, gate, target, sent = setup()
    state = {'ok': True}
    guard.playing = lambda: (state['ok'], 'game menu is open (paused)')
    executor.act('t', 'lane', [Step.key('q')])
    state['ok'] = False
    from addons.chronodivide.executor import NotPlaying
    with pytest.raises(NotPlaying, match='paused'):
        executor.act('t', 'lane', [Step.key('q')])
    assert len(sent) == 1


def test_escape_is_never_sent():
    executor, *_, sent = setup()
    with pytest.raises(DesktopPause, match='blocked'):
        executor.act('t', 'lane', [Step.key('esc')])
    assert sent == []


def test_raw_input_takeover_ignores_injected_input_and_small_jitter():
    from desktop.hooks import RAWINPUT, RIM_TYPEKEYBOARD, RIM_TYPEMOUSE, RawInputWatch
    seen = []
    watch = RawInputWatch.__new__(RawInputWatch)
    watch.on_physical, watch.move_threshold, watch.ignore_keys, watch.motion = seen.append, 12, {0x77}, 0

    def mouse(device, dx=0, dy=0, buttons=0):
        raw = RAWINPUT()
        raw.header.dwType, raw.header.hDevice = RIM_TYPEMOUSE, device
        raw.data.mouse.lLastX, raw.data.mouse.lLastY, raw.data.mouse.usButtonFlags = dx, dy, buttons
        return raw

    def key(device, vk):
        raw = RAWINPUT()
        raw.header.dwType, raw.header.hDevice = RIM_TYPEKEYBOARD, device
        raw.data.keyboard.VKey = vk
        return raw
    watch.process(mouse(None, 500, 500, 1))          # SendInput: never the user
    watch.process(key(None, 0x44))
    watch.process(mouse(7, 3, 2))                    # jitter below threshold
    assert seen == []
    watch.process(mouse(7, 6, 4))                    # accumulated real motion
    watch.process(mouse(7, buttons=1))
    watch.process(key(9, 0x77))                      # F8 resumes, it is not a takeover
    watch.process(key(9, 0x41))
    assert seen == ['mouse-move', 'mouse-button', 'keyboard']


@pytest.mark.parametrize('keys', [('d',), ('l',), ('k',), ('a',), ('delete',), ('enter',), ('tab',), ('ctrl', 'q'),
                                  ('alt', '1'), ('shift', 'd'), ('ctrl', 'p'), ('ctrl', 'shift', '2'), ('ctrl', '0')])
def test_game_keys_outside_the_safe_list_are_refused(keys):
    executor, *_, sent = setup()
    with pytest.raises(DesktopPause, match='blocked'):
        executor.act('t', 'lane', [Step.chord(*keys)])
    assert sent == []


def test_deploy_key_only_while_no_construction_yard_exists():
    executor, guard, *_, sent = setup()
    guard.deploy_allowed = lambda: True
    executor.act('t', 'lane', [Step.key('d')])
    assert len(sent) == 1
    guard.deploy_allowed = lambda: False                  # a yard exists: D would pack it up
    with pytest.raises(DesktopPause, match='blocked'):
        executor.act('t', 'lane', [Step.key('d')])
    assert len(sent) == 1


@pytest.mark.parametrize('keys', [('q',), ('r',), ('h',), ('p',), ('3',), ('space',)])
def test_safe_game_keys_are_sent(keys):
    executor, *_, sent = setup()
    executor.act('t', 'lane', [Step.chord(*keys)])
    assert len(sent) == 1 and len(sent[0]) == 2 * len(keys)


@pytest.mark.parametrize('keys', [('ctrl', '1'), ('shift', '1'), ('ctrl', '2')])
def test_team_keys_hold_the_modifier_in_stages(keys):
    executor, *_, sent = setup()
    executor.act('t', 'lane', [Step.chord(*keys)])
    # modifier down, then the digit down and up, then the modifier up: the game sees it held
    assert [len(batch) for batch in sent] == [1, 2, 1]
    assert [e.data.ki.wVk for e in sent[1]] == [0x30 + int(keys[1])] * 2
    assert all(e.type == 1 for batch in sent for e in batch)


def test_pointer_motion_waits_for_the_game_to_see_a_click_released():
    slept = []
    clock = [10.0]

    def sleep(seconds):
        slept.append(round(seconds, 3))
        clock[0] += seconds
    executor, *_, sent = setup(sleep=sleep)
    executor.clock = lambda: clock[0]
    executor.act('t', 'lane', [Step.click((425, 650), 'right'), Step.key('h'), Step.move((500, 400))])
    # the click rests on its point before the press (the game reads what is under the pointer);
    # the key goes at once; the move waits until 60 ms after the click (right-drag would scroll)
    assert [len(batch) for batch in sent] == [1, 2, 2, 1]
    assert slept[0] == 0.05 and abs(sum(slept[1:]) - 0.06) < 0.02
