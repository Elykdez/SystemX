from types import SimpleNamespace
from pathlib import Path
import json
import sys

import numpy as np
import pytest

if sys.platform != 'win32':
    pytest.skip('Windows desktop addon', allow_module_level=True)
cv2 = pytest.importorskip('cv2', reason='Install optional desktop addon dependencies to test vision')
pytest.importorskip('mss')

from desktop.input import Gate, absolute_point
from desktop.target import DesktopPause, Geometry
from desktop.capture import Frame
from addons.minesweeper.board import Board, Box, Cell
from addons.minesweeper.desktop_driver import DesktopDriver
from addons.minesweeper.session import MinesweeperSession
from addons.minesweeper.visual.recognition import Grid, Profile, recognize


def test_absolute_coordinates_across_negative_origin_and_scaling():
    for scale in (1, 1.25, 1.5, 2):
        width, height = round(1920 * scale), round(1080 * scale)
        assert absolute_point(-width, 0, (-width, 0, width * 2, height)) == (0, 0)
        assert absolute_point(width - 1, height - 1, (-width, 0, width * 2, height)) == (65535, 65535)
        with pytest.raises(DesktopPause):
            absolute_point(width, 0, (-width, 0, width * 2, height))


def test_pause_resume_rejects_delayed_decision_from_previous_run():
    gate = Gate()
    gate.start()
    old = gate.generation
    gate.pause()
    gate.start()
    assert not gate.allowed(old)
    assert gate.allowed(gate.generation)
    gate.stop()
    gate.start()
    assert not gate.allowed(gate.generation)


def test_grid_rejects_bad_dimensions_and_clipping():
    with pytest.raises(ValueError):
        Grid(0, 0, 9, 9, 9, 9)
    grid = Grid(0, 0, 40, 40, 2, 2)
    with pytest.raises(ValueError, match='clipped'):
        grid.crop(np.zeros((20, 20, 3), np.uint8), list(grid.boxes())[-1][2])


def test_unknown_and_ambiguous_samples_never_become_empty_cells():
    profile = Profile()
    black = np.zeros((32, 32, 3), np.uint8)
    assert profile.classify(black)[0] is None
    profile.add('0', black)
    assert profile.classify(np.full_like(black, 255))[0] is None
    profile.add('hidden', black)
    assert profile.classify(black)[0] is None
    board, details = recognize(black, Grid(0, 0, 32, 32, 1, 1), profile)
    assert board is None and details[0]['label'] is None


def test_profile_roundtrip_and_untrusted_data_validation(tmp_path):
    p = Profile()
    p.add('1', np.zeros((32, 32, 3), np.uint8))
    path = tmp_path / 'profile.json'
    p.save(path)
    assert Profile.load(path).classify(np.zeros((32, 32, 3), np.uint8))[0] == '1'
    path.write_text('{"version":1,"name":"x","threshold":NaN,"margin":0.1}')
    with pytest.raises(ValueError):
        Profile.load(path)


def board(states=('hidden', 'hidden')):
    return Board(1, 2, tuple(Cell(0, c, state, Box(c * 20, 0, 20, 20)) for c, state in enumerate(states)), 'playing')


def test_cancelled_model_response_never_clicks():
    gate = Gate()
    gate.start()
    generation = gate.generation
    clicks = []
    driver = SimpleNamespace(observe=board, click=lambda *args: clicks.append(args))
    class Brain:
        def choose(self, current, options):
            gate.pause()
            gate.start()
            return options[0], {}
    session = MinesweeperSession(driver, Brain(), can_act=lambda: gate.allowed(generation))
    assert session.step()['status'] == 'paused'
    assert not clicks


def test_changed_board_immediately_before_input_is_rejected():
    driver = object.__new__(DesktopDriver)
    driver.last_board = board()
    driver.last_frame = object()
    driver.observe = lambda: board(('0', 'hidden'))
    with pytest.raises(DesktopPause, match='changed'):
        driver.click(driver.last_board.cells[0], 'left')


def test_geometry_change_rejects_observation_before_recognition():
    g = Geometry(0, 0, 40, 20, 96)
    frame = Frame(1, 0, Geometry(1, 0, 40, 20, 96), np.zeros((20, 40, 3), np.uint8))
    capture = SimpleNamespace(read=lambda region: frame)
    driver = DesktopDriver(capture, None, Grid(0, 0, 40, 20, 1, 2), Profile(), Gate(), 0, g)
    with pytest.raises(DesktopPause, match='Geometry'):
        driver._read()


def test_click_result_requires_selected_cell_change(monkeypatch):
    driver = object.__new__(DesktopDriver)
    driver.pending = (0, 0, 'left')
    driver.gate = Gate()
    driver.gate.start()
    driver.generation = driver.gate.generation
    driver.observe = lambda: board(('hidden', '0'))
    ticks = iter([0, 0, 4])
    monkeypatch.setattr('addons.minesweeper.desktop_driver.time.monotonic', lambda: next(ticks))
    monkeypatch.setattr('addons.minesweeper.desktop_driver.time.sleep', lambda _: None)
    with pytest.raises(DesktopPause, match='no repeat'):
        driver.wait_for_change(board())
    assert driver.pending is None


def test_flag_postcondition_accepts_only_a_flag():
    driver = object.__new__(DesktopDriver)
    driver.pending = (0, 0, 'right')
    driver.gate = Gate()
    driver.gate.start()
    driver.generation = driver.gate.generation
    driver.observe = lambda: board(('flag', 'hidden'))
    assert driver.wait_for_change(board()).cells[0].state == 'flag'


@pytest.mark.parametrize('name', ['heldout-win', 'loss', 'supervised-win'])
def test_recorded_visual_boards(name):
    folder = Path(__file__).parent / 'fixtures/visual'
    expected = json.loads((folder / f'{name}.json').read_text())
    image = cv2.imread(str(folder / f'{name}.png'))
    profile = Profile.load(Path(__file__).parents[1] / 'visual/profiles/classic-web.json')
    actual, details = recognize(image, Grid(0, 0, image.shape[1], image.shape[0],
                                          expected['rows'], expected['cols'], expected['mines']), profile)
    assert actual is not None, details
    assert actual.status == expected['status']
    assert [[c.row, c.col, c.state] for c in actual.cells] == expected['cells']


def test_unfamiliar_digit_shape_is_rejected_instead_of_guessed():
    profile = Profile.load(Path(__file__).parents[1] / 'visual/profiles/classic-web.json')
    image = np.full((36, 36, 3), 192, np.uint8)
    cv2.putText(image, '8', (9, 25), cv2.FONT_HERSHEY_SIMPLEX, .8, (80, 80, 80), 2)
    assert profile.classify(image)[0] is None


def test_occlusion_blocks_mouse_without_dispatch(monkeypatch):
    from desktop.input import Mouse
    from desktop import input as input_module
    gate = Gate()
    gate.start()
    geometry = Geometry(0, 0, 200, 200, 96)
    def obscured(*args, **kwargs):
        raise DesktopPause('covered')
    monkeypatch.setattr(input_module, 'cursor', lambda: (0, 0))
    target = SimpleNamespace(geometry=lambda: geometry, check_point=obscured)
    sent = []
    monkeypatch.setattr(input_module.user32, 'SendInput', lambda *args: sent.append(args))
    with pytest.raises(DesktopPause, match='covered'):
        Mouse(target, gate).click(10, 10, 'left', geometry, gate.generation)
    assert sent == []


def test_partial_input_failure_releases_button_and_stops(monkeypatch):
    from desktop import input as input_module
    gate = Gate()
    gate.start()
    geometry = Geometry(0, 0, 200, 200, 96)
    target = SimpleNamespace(geometry=lambda: geometry, check_point=lambda *a, **kw: None)
    monkeypatch.setattr(input_module, 'cursor', lambda: (0, 0))
    monkeypatch.setattr(input_module.user32, 'GetAsyncKeyState', lambda key: 0)
    monkeypatch.setattr(input_module.user32, 'GetSystemMetrics', lambda key: {76:0,77:0,78:1920,79:1080}[key])
    calls = []
    def send(count, *args):
        calls.append(count)
        return 2 if count == 3 else count
    monkeypatch.setattr(input_module.user32, 'SendInput', send)
    with pytest.raises(DesktopPause, match='complete click'):
        input_module.Mouse(target, gate).click(10, 10, 'left', geometry, gate.generation)
    assert calls == [3, 1]


def test_initial_pointer_parking_sends_only_movement(monkeypatch):
    from desktop import input as input_module
    import ctypes
    gate = Gate()
    gate.start()
    geometry = Geometry(0, 0, 200, 200, 96)
    target = SimpleNamespace(geometry=lambda: geometry, check_point=lambda *a, **kw: None)
    monkeypatch.setattr(input_module, 'cursor', lambda: (0, 0))
    monkeypatch.setattr(input_module.user32, 'GetAsyncKeyState', lambda key: 0)
    monkeypatch.setattr(input_module.user32, 'GetSystemMetrics', lambda key: {76:0,77:0,78:1920,79:1080}[key])
    events = []
    def send(count, pointer, size):
        event = ctypes.cast(pointer, ctypes.POINTER(input_module.Input)).contents
        events.append(event.data.mi.dwFlags)
        return count
    monkeypatch.setattr(input_module.user32, 'SendInput', send)
    mouse = input_module.Mouse(target, gate)
    mouse.move(50, 10, geometry, gate.generation)
    assert events == [0xC001]
    generation = gate.generation
    gate.pause()
    with pytest.raises(DesktopPause, match='cancelled'):
        mouse.move(50, 10, geometry, generation)
    assert events == [0xC001]


def test_prepare_parks_outside_board_before_recognition(monkeypatch):
    calls = []
    geometry = Geometry(100, 50, 600, 500, 96)
    grid = Grid(80, 100, 180, 180, 9, 9, 10)
    mouse = SimpleNamespace(move=lambda *args: calls.append(args))
    driver = DesktopDriver(None, mouse, grid, Profile(), Gate(), 12, geometry)
    monkeypatch.setattr('addons.minesweeper.desktop_driver.time.sleep', lambda _: None)
    driver.prepare()
    assert calls == [(270, 130, geometry, 12)]


def test_classification_cache_reuses_pixels_but_invalidates_on_teaching(monkeypatch):
    from addons.minesweeper.visual import recognition
    profile = Profile()
    cell = np.full((36, 36, 3), 192, np.uint8)
    profile.add('0', cell)
    calls = []
    real_features = recognition.features
    monkeypatch.setattr(recognition, 'features', lambda image: (calls.append(1), real_features(image))[1])
    first = profile.classify(cell)
    assert profile.classify(cell.copy()) == first
    assert len(calls) == 1
    profile.add('hidden', cell)
    assert profile.classify(cell)[0] is None  # The newly taught sample makes this ambiguous.
    assert len(calls) == 3


def test_classification_cache_checks_changed_pixels_and_thresholds():
    profile = Profile.load(Path(__file__).parents[1] / 'visual/profiles/classic-web.json')
    known = profile.samples['1'][0].copy()
    assert profile.classify(known)[0] == '1'
    unknown = np.zeros((32, 32, 3), np.uint8)
    assert profile.classify(unknown)[0] is None
    profile.margin = 2
    assert profile.classify(known)[0] is None
    profile.margin = .012
    assert profile.classify(known)[0] == '1'


def test_classification_cache_has_bounded_memory():
    profile = Profile()
    profile.add('0', np.full((32, 32, 3), 192, np.uint8))
    random = np.random.default_rng(42)
    for _ in range(520):
        profile.classify(random.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    assert len(profile._classifications) == 512


def test_fast_observe_still_requires_two_matching_frames(monkeypatch):
    driver = object.__new__(DesktopDriver)
    driver._previous_observation = None
    frames = iter([board(), board(('0', 'hidden')), board(('0', 'hidden'))])
    driver._read = lambda: next(frames)
    delays, clock = [], [0.0]
    def sleep(delay):
        delays.append(delay)
        clock[0] += delay
    monkeypatch.setattr('addons.minesweeper.desktop_driver.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('addons.minesweeper.desktop_driver.time.sleep', sleep)
    assert driver.observe().cells[0].state == '0'
    assert delays == pytest.approx([.02, .02])


def test_window_picker_excludes_tool_overlays(monkeypatch):
    from desktop import target
    monkeypatch.setattr(target.user32, 'EnumWindows', lambda visit, _: [visit(hwnd, 0) for hwnd in (1, 2)])
    monkeypatch.setattr(target.user32, 'IsWindowVisible', lambda _: True)
    monkeypatch.setattr(target.user32, 'GetWindowLongPtrW', lambda hwnd, _: 0x80 if hwnd == 1 else 0)
    monkeypatch.setattr(target.Target, 'attach', lambda hwnd: SimpleNamespace(hwnd=hwnd, title='Game', pid=-1))
    assert [window.hwnd for window in target.windows()] == [2]


@pytest.fixture
def flag_redraw(monkeypatch):
    from dataclasses import replace
    from addons.minesweeper import desktop_driver
    image = cv2.imread(str(Path(__file__).parent / 'fixtures/visual/flag-redraw.png'))
    grid = Grid(0, 0, 324, 324, 9, 9, 10)
    profile = Profile.load(Path(__file__).parents[1] / 'visual/profiles/classic-web.json')
    transient, _ = recognize(image, grid, profile)
    assert transient.cells[34].state == '0'  # Real frame before the flag glyph was painted.
    def with_state(state):
        return replace(transient, cells=tuple(replace(c, state=state) if (c.row, c.col) == (3, 7)
                                             else c for c in transient.cells))
    previous, flagged = with_state('hidden'), with_state('flag')
    geometry = Geometry(0, 0, 324, 324, 96)
    capture = SimpleNamespace(read=lambda region: Frame(1, 0, geometry, image),
                              target=SimpleNamespace(check_point=lambda *a: None))
    gate = Gate()
    gate.start()
    driver = DesktopDriver(capture, None, grid, profile, gate, gate.generation, geometry)
    driver.pending = (3, 7, 'right')
    clock = [0.0]
    monkeypatch.setattr(desktop_driver.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(desktop_driver.time, 'sleep', lambda delay: clock.__setitem__(0, clock[0] + delay))
    return driver, previous, transient, flagged


def test_flag_redraw_retries_contradictory_frames_then_confirms(flag_redraw, monkeypatch):
    driver, previous, transient, flagged = flag_redraw
    frames = iter([transient, transient, flagged, flagged])
    current = [None]
    read = driver.capture.read
    def redraw(region):
        from dataclasses import replace
        current[0] = next(frames)
        frame = read(region)
        pixels = frame.pixels.copy()
        if current[0] == flagged:
            pixels[126, 270] = [0, 0, 255]  # Signal the glyph redraw to the pixel cache.
        return replace(frame, pixels=pixels)
    driver.capture.read = redraw
    monkeypatch.setattr('addons.minesweeper.desktop_driver.recognize', lambda *a: (current[0], []))
    assert driver.wait_for_change(previous) == flagged
    assert driver.pending is None
    assert driver.gate.allowed(driver.generation)


def test_persistent_contradiction_times_out_without_repeating_click(flag_redraw, monkeypatch):
    driver, previous, transient, _ = flag_redraw
    monkeypatch.setattr('addons.minesweeper.desktop_driver.recognize', lambda *a: (transient, []))
    with pytest.raises(DesktopPause, match='Click result unconfirmed; no repeat sent.*contradict'):
        driver.wait_for_change(previous)
    assert driver.pending is None


def test_contradictory_board_before_input_still_blocks_click(flag_redraw, monkeypatch):
    driver, previous, transient, _ = flag_redraw
    monkeypatch.setattr('addons.minesweeper.desktop_driver.recognize', lambda *a: (transient, []))
    driver.last_board = previous
    driver.last_frame = object()
    with pytest.raises(DesktopPause, match='contradict'):
        driver.click(previous.cells[34], 'right')


@pytest.fixture
def stable_observer(monkeypatch):
    clock = [0.0]
    reads = []
    driver = DesktopDriver(None, None, Grid(0, 40, 40, 20, 1, 2), Profile(), Gate(), 0,
                           Geometry(0, 0, 100, 100, 96))
    def read():
        reads.append(clock[0])
        return board()
    driver._read = read
    monkeypatch.setattr('addons.minesweeper.desktop_driver.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('addons.minesweeper.desktop_driver.time.sleep',
                        lambda delay: clock.__setitem__(0, clock[0] + delay))
    return driver, clock, reads


def test_recent_stable_frame_still_requires_a_new_capture(stable_observer):
    driver, clock, reads = stable_observer
    driver.observe()
    clock[0] += .06  # Model work already supplied the gap between frames.
    driver.observe()
    assert reads == pytest.approx([0, .02, .08])


def test_back_to_back_checks_preserve_minimum_frame_gap(stable_observer):
    driver, _, reads = stable_observer
    driver.observe()
    driver.observe()
    assert reads == pytest.approx([0, .02, .04])


def test_expired_stability_requires_two_new_captures(stable_observer):
    driver, clock, reads = stable_observer
    driver.observe()
    clock[0] += .11
    driver.observe()
    assert reads == pytest.approx([0, .02, .13, .15])


def test_changed_frame_cannot_reuse_previous_stability(stable_observer):
    driver, clock, reads = stable_observer
    driver.observe()
    clock[0] += .03
    original = driver._read
    def changed():
        original()
        return board(('0', 'hidden'))
    driver._read = changed
    assert driver.observe().cells[0].state == '0'
    assert reads == pytest.approx([0, .02, .05, .07])


def test_capture_failure_discards_previous_stability(stable_observer):
    driver, _, reads = stable_observer
    driver.observe()
    original = driver._read
    def fail():
        raise DesktopPause('covered')
    driver._read = fail
    with pytest.raises(DesktopPause, match='covered'):
        driver.observe()
    driver._read = original
    driver.observe()
    assert len(reads) == 4


def test_click_discards_stability_before_verifying_result(stable_observer):
    driver, _, reads = stable_observer
    driver.gate.start()
    driver.generation = driver.gate.generation
    driver.last_frame = object()
    clicks = []
    driver.mouse = SimpleNamespace(click=lambda *a, **kw: clicks.append(a))
    current = driver.observe()
    driver.click(current.cells[0], 'left')
    assert len(clicks) == 1
    assert driver._previous_observation is None
    driver.observe()
    assert len(reads) == 5  # Two initially, one before input, two after input.


@pytest.mark.parametrize('region, origin, shape', [(None, (0, 0), (300, 400, 3)),
                                                  ((80, 50, 100, 120), (80, 50), (120, 100, 3))])
def test_capture_region_uses_physical_coordinates(region, origin, shape):
    from desktop.capture import Capture
    calls = []
    geometry = Geometry(-500, 100, 400, 300, 144)
    capture = object.__new__(Capture)
    capture.sequence = 0
    capture.target = SimpleNamespace(geometry=lambda: geometry, check_point=lambda *a, **kw: None)
    def grab(rect):
        calls.append(rect)
        return np.zeros((rect['height'], rect['width'], 4), np.uint8)
    capture.source = SimpleNamespace(monitors=[dict(left=-1920, top=0, width=3840, height=1080)], grab=grab)
    frame = capture.read(region)
    assert frame.geometry == geometry and frame.origin == origin
    assert frame.pixels.shape == shape
    assert calls == [dict(left=geometry.x + origin[0], top=geometry.y + origin[1],
                          width=shape[1], height=shape[0])]
    with pytest.raises(DesktopPause, match='outside'):
        capture.read((350, 0, 100, 100))
    assert len(calls) == 1


def test_cropped_recognition_preserves_window_coordinates():
    image = cv2.imread(str(Path(__file__).parent / 'fixtures/visual/heldout-win.png'))
    profile = Profile.load(Path(__file__).parents[1] / 'visual/profiles/classic-web.json')
    expected, _ = recognize(image, Grid(0, 0, 324, 324, 9, 9, 10), profile)
    actual, _ = recognize(image, Grid(150, 220, 324, 324, 9, 9, 10), profile, (150, 220))
    assert [c.state for c in actual.cells] == [c.state for c in expected.cells]
    assert actual.cells[0].box.center == (168, 238)
    assert actual.cells[-1].box.center == (456, 526)


def test_identical_board_skips_recognition_but_keeps_capture_and_hit_checks(monkeypatch):
    from addons.minesweeper import desktop_driver
    image = cv2.imread(str(Path(__file__).parent / 'fixtures/visual/heldout-win.png'))
    grid = Grid(150, 220, 324, 324, 9, 9, 10)
    geometry = Geometry(100, 200, 700, 800, 96)
    profile = Profile.load(Path(__file__).parents[1] / 'visual/profiles/classic-web.json')
    captures, hits, notifications, recognitions = [], [], [], []
    frame = Frame(1, 0, geometry, image, (150, 220))
    def read(region):
        captures.append(region)
        return frame
    target = SimpleNamespace(check_point=lambda *a: hits.append(a))
    capture = SimpleNamespace(read=read, target=target)
    driver = DesktopDriver(capture, None, grid, profile, Gate(), 0, geometry,
                           notify=lambda *a: notifications.append(a))
    def recognize_once(*args):
        recognitions.append(1)
        return recognize(*args)
    monkeypatch.setattr(desktop_driver, 'recognize', recognize_once)
    driver._read()
    driver._read()
    assert len(recognitions) == len(notifications) == 1
    assert captures == [(150, 220, 324, 324)] * 2
    assert len(hits) == 162 and hits[0] == (268, 438)
    image[0, 0, 0] ^= 1
    driver._read()
    assert len(recognitions) == 2
    profile.threshold = .074
    driver._read()
    assert len(recognitions) == 3
    def covered(*args):
        raise DesktopPause('covered')
    target.check_point = covered
    with pytest.raises(DesktopPause, match='covered'):
        driver._read()
    assert len(recognitions) == 3
