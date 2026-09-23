"""The setup window, driven like a user would, with captures from recorded frames (no real windows)."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from desktop.target import Geometry, Target

tk = pytest.importorskip('tkinter')
FIXTURES = Path(__file__).with_name('fixtures')


def page(name, at=(35, 75)):
    client = np.full((887, 1094, 3), 60, np.uint8)
    client[:31] = (240, 240, 240)
    image = cv2.imread(str(FIXTURES / name))
    client[at[1]:at[1] + 768, at[0]:at[0] + 1024] = image
    return client


def test_capture_find_check_teach_and_use(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('No display for Tk')
    root.withdraw()
    from addons.chronodivide import calibration, client, setup_ui
    from addons.chronodivide.calibration import Candidate, Setup
    bundled = client.ClientProfile.load('chronodivide')
    monkeypatch.setattr(client, 'CLIENTS', tmp_path / 'clients')        # taught profiles are saved here
    monkeypatch.setattr(setup_ui, 'save_setup', lambda setup: calibration.save_setup(setup, tmp_path / 'setup.json'))
    monkeypatch.setattr(setup_ui.simpledialog, 'askstring', lambda *a, **k: 'My CD')
    game = Target(7, 70, 1, 'Chrono Divide')
    monkeypatch.setattr(setup_ui, 'rank_windows', lambda profile: [Candidate(game, 'chrome.exe', 6, True)])
    frames = [page('frame-play.png')]
    capture = lambda target: SimpleNamespace(pixels=frames[0], geometry=Geometry(0, 0, 1094, 887, 96))
    done = []
    window = setup_ui.SetupWindow(root, bundled, Setup(), capture=capture,
                                  on_done=lambda profile, setup, target: done.append((profile, setup, target)))
    try:
        window.top.withdraw()
        window.capture()                                           # captured, and the view found at once
        assert window.rect.tuple() == (35, 75, 1024, 768)
        assert window.status.get().startswith('Game view 1024x768') and 'HUD' in window.status.get()
        assert sum(r.item is not None for r in window.readings) >= 2
        # Teach: the credits, a cameo (the first slot is the Tesla Reactor), our color from a unit.
        window.credits.set('5200')
        window.teach_credits()
        window.select_slot(877 + 30, 228 + 20)
        assert window.selected == 0
        window.item.set('tesla_reactor')
        window.teach_cameo()
        assert {'digits', 'cameos'} <= window.taught
        with pytest.raises(ValueError):
            window.credits.set('52x')
            window.teach_credits()
        window.save_profile()                                      # a bundled profile: saved as a new one
        assert window.profile.slug == 'my-cd' and (tmp_path / 'clients' / 'my-cd.json').exists()
        assert (tmp_path / 'clients' / 'my-cd' / 'cameos.npz').exists() and not window.taught
        saved = client.ClientProfile.load(tmp_path / 'clients' / 'my-cd.json')
        assert 'tesla_reactor' in saved.cameos().templates and saved.input == 'locked'
        assert saved.templates['hud'] == bundled.templates['hud']      # untaught: still the bundled file
        window.use()
        profile, setup, target = done[0]
        assert target.hwnd == 7 and setup.view == [35, 75, 1024, 768] and setup.client == [1094, 887]
        assert (tmp_path / 'setup.json').exists()
    finally:
        try:
            window.top.destroy()
        except tk.TclError:
            pass
        root.destroy()


def test_a_dragged_view_is_used_when_no_sidebar_is_found(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('No display for Tk')
    root.withdraw()
    from addons.chronodivide import setup_ui
    from addons.chronodivide.calibration import Candidate, Setup
    game = Target(7, 70, 1, 'Chrono Divide')
    monkeypatch.setattr(setup_ui, 'rank_windows', lambda profile: [Candidate(game, 'chrome.exe', 6, True)])
    capture = lambda target: SimpleNamespace(pixels=page('frame-setup.png'), geometry=Geometry(0, 0, 1094, 887, 96))
    window = setup_ui.SetupWindow(root, setup=Setup(), capture=capture)
    try:
        window.top.withdraw()
        window.capture()
        assert window.rect is None and 'No match sidebar' in window.status.get()
        s = window.scale
        window.press(SimpleNamespace(x=35 * s, y=75 * s))
        window.release(SimpleNamespace(x=(35 + 1024) * s, y=(75 + 768) * s))
        assert abs(window.rect.x - 35) <= 2 and abs(window.rect.width - 1024) <= 3
        assert 'Not in a match' in window.status.get()
    finally:
        window.top.destroy()
        root.destroy()
