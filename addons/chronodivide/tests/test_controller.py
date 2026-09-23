"""Controller window with a fake player: renders preview, status, and panels (window withdrawn)."""

from types import SimpleNamespace
import time

import numpy as np
import pytest

from addons.chronodivide.perception.scene import Track

tk = pytest.importorskip('tkinter')


class FakePlayer:
    def __init__(self):
        self.preview = np.zeros((768, 1024, 3), np.uint8)
        self.armed = self.paused = self.armed_on_start = 0
        self.output = 'run-folder'
        self.last_playing = (False, 'not in a match')
        now = time.monotonic()
        self.units = [Track(1, (400, 300, 440, 330), 'ours', 'vehicle', 'bars', now, now, 'rhino_tank', True, 0.8),
                      Track(2, (600, 200, 620, 220), 'enemy', 'unit', 'color', now, now)]
        self.items = {'rhino_tank': {'name': 'Rhino Tank', 'short': 'Rhino'}}

    def arm_on_start(self):
        self.armed_on_start += 1

    def arm(self):
        self.armed += 1

    def pause(self, reason=''):
        self.paused += 1

    def stop(self):
        pass

    def view_state(self):
        lane = dict(status='building', item='tesla_reactor', progress=0.5, available=['tesla_reactor'], age=0.1)
        idle = dict(status='unknown', item=None, progress=None, available=[], age=None)
        return dict(status=dict(state='acting', reason=''), mode='execute', policy='systemone', hud=getattr(self, 'hud', None),
                    playing=(True, 'playing'), plan='queue:tesla_reactor',
                    world=dict(mode='hud', credits=5200, reserved=0, power=dict(level='ok'), tab='buildings',
                               structures=dict(construction_yard=1), units={},
                               lanes=dict(buildings=lane, defenses=idle, infantry=idle, vehicles=idle)),
                    intent=dict(objective='build_army', attack_size=None, production={}, provenance={}),
                    tasks=dict(active={}, background=0, recent=[]), systemone=dict(strategy=dict(systemone=3)),
                    counts=dict(transactions=4, events=12, commands=4), apm=48.0,
                    timings_ms=dict(capture=6.0, perceive=4.0, act=1.0), events=[dict(t=1.0, kind='deployed')],
                    pointer=dict(position=(900, 400), gain=1.0),
                    units=dict(ours={'Rhino': 1}, health=0.8, patches=2, enemy=dict(unit=1, building=0),
                               tracked=2, ms=31.0, scans=12, shift=(0, 0), error=None))


def test_controller_renders_player_state_and_routes_hotkeys():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('No display for Tk')
    root.withdraw()
    from addons.chronodivide.controller import Controller
    controller = Controller(root, SimpleNamespace(policy='systemone', server='http://127.0.0.1:8000', record_fps=0))
    try:
        controller.player = FakePlayer()
        controller.refresh()
        text = controller.text.get('1.0', 'end')
        assert 'credits 5200' in text and 'tesla_reactor' in text and 'APM 48.0' in text
        assert 'SCENE ours: 1 Rhino' in text and 'unconfirmed color patches: 2 ours' in text
        assert 'scan 31.0 ms' in text
        assert controller.state.get().startswith('ACTING') and 'queue:tesla_reactor' in controller.state.get()
        assert controller.photo is not None
        controller.player.view = SimpleNamespace(error='')
        controller.refresh()
        assert controller.window_hint.get() == ''
        assert controller.window_warning.winfo_manager() == ''
        controller.player.view.error = 'The game canvas is clipped. Enlarge the browser.'
        controller.refresh()
        assert 'Cannot proceed' in controller.window_hint.get()
        assert 'canvas is clipped' in controller.window_hint.get()
        assert controller.window_warning.winfo_manager() == 'pack'
        controller.player.view.error = ''
        controller.refresh()
        assert controller.window_hint.get() == ''
        assert controller.window_warning.winfo_manager() == ''
        controller.player.hud = dict(mode='other', scores={'top_buttons': .52, 'sidebar_bottom': .69}, required=.7)
        controller.refresh()
        assert '52%' in controller.window_hint.get() and '69%' in controller.window_hint.get()
        assert 'required 70%' in controller.window_hint.get() and 'input is stopped' in controller.window_hint.get()
        controller.player.hud['mode'] = 'hud'
        controller.refresh()
        assert controller.window_hint.get() == ''
        controller.hotkey('run')
        controller.act()
        controller.hotkey('pause')
        assert (controller.player.armed_on_start, controller.player.paused) == (2, 1)
    finally:
        controller.player = None
        controller.close()
