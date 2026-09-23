"""Controller workflow checks with a withdrawn Tk window and no OS input."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

if sys.platform != 'win32':
    pytest.skip('Windows desktop addon', allow_module_level=True)
cv2 = pytest.importorskip('cv2')
pytest.importorskip('mss')
pytest.importorskip('PIL')
tk = pytest.importorskip('tkinter')

from desktop.capture import Frame
from desktop.target import Target, Geometry
from addons.minesweeper import desktop_player as player
from addons.minesweeper.visual.recognition import Grid, Profile


@pytest.fixture
def controller(monkeypatch):
    import numpy as np
    image = np.full((430, 400, 3), 160, np.uint8)
    crop = cv2.imread(str(Path(__file__).parent / 'fixtures/visual/heldout-win.png'))
    image[70:394, 30:354] = crop
    frame = Frame(1, 0, Geometry(0, 0, 400, 430, 96), image)
    target = Target(99, 1, 2, 'Fixture game')
    monkeypatch.setattr(player, 'windows', lambda: [target])
    monkeypatch.setattr(player, 'Capture', lambda _: SimpleNamespace(read=lambda: frame, close=lambda: None))
    monkeypatch.setattr(player, 'Hotkeys', lambda callback: SimpleNamespace(close=lambda: None))
    from desktop.target import user32
    monkeypatch.setattr(user32, 'SetForegroundWindow', lambda _: True)
    root = tk.Tk()
    root.withdraw()
    app = player.Controller(root, SimpleNamespace(policy='logic', server='http://127.0.0.1:8000'))
    try:
        yield app
    finally:
        app.close()


def test_capture_drag_calibration_and_preview(controller):
    controller.capture()
    controller.press(SimpleNamespace(x=30, y=70))
    controller.release(SimpleNamespace(x=354, y=394))
    assert controller.grid == Grid(30, 70, 324, 324, 9, 9, 10)
    assert all(d['label'] is not None for d in controller.details)
    assert '0 unknown cells' in controller.status.get()
    assert len(controller.canvas.find_all()) == 1 + 81 * 2


def test_teaching_profile_persistence_and_calibration(controller, tmp_path, monkeypatch):
    from tkinter import filedialog
    controller.capture()
    controller.grid = Grid(30, 70, 324, 324, 9, 9, 10)
    controller.new_profile()
    assert all(d['label'] is None for d in controller.details)
    controller.press(SimpleNamespace(x=48, y=88))
    controller.release(SimpleNamespace(x=48, y=88))
    controller.label.set('0')
    controller.teach()
    assert '0' in controller.profile.samples
    path = tmp_path / 'calibration.json'
    monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **kw: str(path))
    controller.save_config()
    saved = json.loads(path.read_text())
    assert saved['target']['hwnd'] == 99
    assert saved['grid']['rows'] == 9
    assert Path(saved['profile']).is_file()
    monkeypatch.setattr(filedialog, 'askopenfilename', lambda **kw: saved['profile'])
    controller.profile = Profile()
    controller.load_profile()
    assert '0' in controller.profile.samples


def test_recalibration_clears_old_grid_even_when_capture_fails(controller, monkeypatch):
    controller.capture()
    controller.grid = Grid(30, 70, 324, 324, 9, 9, 10)
    def failed_capture(_):
        raise RuntimeError('Capture unavailable')
    monkeypatch.setattr(player, 'Capture', failed_capture)
    with pytest.raises(RuntimeError, match='unavailable'):
        controller.capture()
    assert controller.frame is None and controller.grid is None


def test_pause_hotkey_invalidates_pending_generation(controller):
    controller.gate.start()
    generation = controller.gate.generation
    controller.hotkey('pause')
    assert not controller.gate.allowed(generation)
    assert controller.events.get_nowait() == ('command', 'pause')


def test_board_only_preview_retains_calibration_and_teaching_coordinates(controller):
    from dataclasses import replace
    controller.capture()
    original = controller.frame
    controller.frame = replace(original, pixels=original.pixels[70:394, 30:354].copy(), origin=(30, 70))
    controller.press(SimpleNamespace(x=0, y=0))
    controller.release(SimpleNamespace(x=324, y=324))
    assert controller.grid == Grid(30, 70, 324, 324, 9, 9, 10)
    assert '0 unknown cells' in controller.status.get()
    controller.press(SimpleNamespace(x=18, y=18))
    controller.release(SimpleNamespace(x=18, y=18))
    assert controller.selected[:2] == (0, 0)
    assert controller.selected[2].center == (48, 88)
    controller.profile = Profile()
    controller.label.set('0')
    controller.teach()
    assert controller.details[0]['label'] == '0'
