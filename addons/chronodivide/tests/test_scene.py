"""The inspector's scene: detections from recorded frames, tracks that follow the camera and keep
what they learned, fading, and the worker thread that keeps scans off the control loop."""

from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np

from addons.chronodivide.perception.scene import (CameraShift, SceneWorker, Seen, Tracker, draw_scene,
                                                  scan, summary)
from addons.chronodivide.world import load_knowledge

FIXTURES = Path(__file__).with_name('fixtures')
TACTICAL = (0, 0, 850, 736)
ITEMS = load_knowledge()['items']


def frame(name):
    return cv2.imread(str(FIXTURES / name))


def moved(canvas, dx, dy):
    """The tactical view as after a camera move: the ground slides by (dx, dy)."""
    out = canvas.copy()
    out[:736, :850] = cv2.warpAffine(canvas[:736, :850], np.float32([[1, 0, dx], [0, 1, dy]]), (850, 736))
    return out


def test_color_alone_never_counts_as_a_building_or_unit():
    canvas = np.zeros((768, 1024, 3), np.uint8)
    canvas[200:270, 100:180] = (0, 0, 220)  # bigger than the old building threshold
    canvas[200:230, 400:420] = (220, 110, 35)
    seen = scan(canvas, TACTICAL)
    assert seen and all(s.kind == 'patch' and s.source == 'color' for s in seen)
    counts = summary(Tracker().update(seen, 0), now=0)
    assert counts['ours'] == {} and counts['enemy'] == dict(unit=0, building=0)
    assert counts['patches'] == counts['enemy_patches'] == 1


def test_recognized_yard_has_one_full_box_and_claims_its_color_patches():
    seen = scan(frame('army-selected.png'), TACTICAL, {'rhino_tank': 20}, ITEMS)
    yards = [s for s in seen if s.name == 'construction_yard']
    assert len(yards) == 1 and yards[0].source == 'template'
    x0, y0, x1, y1 = yards[0].box
    assert not any(s.source == 'color' and x0 <= s.anchor[0] < x1 and y0 <= s.anchor[1] < y1 for s in seen)


def test_detector_confirms_large_patches_and_counts_enemy_vehicles():
    canvas = np.zeros((768, 1024, 3), np.uint8)
    canvas[200:255, 100:160] = (0, 0, 220)
    canvas[300:355, 400:460] = (220, 110, 35)
    detector = SimpleNamespace(detect=lambda *_: [
        SimpleNamespace(box=(95, 195, 165, 260), side='ours', kind='vehicle'),
        SimpleNamespace(box=(395, 295, 465, 360), side='enemy', kind='vehicle')])
    seen = scan(canvas, TACTICAL, detector=detector)
    assert len(seen) == 2 and all(s.source == 'detector' for s in seen)
    counts = summary(Tracker().update(seen, 0), now=0)
    assert counts['ours'] == {'vehicle': 1} and counts['enemy'] == dict(unit=1, building=0)
    assert counts['patches'] == counts['enemy_patches'] == 0


def test_a_scan_sees_our_selected_units_our_other_objects_and_the_enemy():
    built = {'rhino_tank': 20, 'flak_track': 3, 'tesla_trooper': 8}
    seen = scan(frame('army-selected.png'), TACTICAL, built, ITEMS)
    bars = [s for s in seen if s.source == 'bars']
    assert len(bars) >= 15 and any(s.name == 'tesla_trooper' for s in bars)
    assert any(s.side == 'ours' and s.kind == 'building' for s in seen)
    enemy = scan(frame('enemy-base-siege.png'), TACTICAL)
    assert any(s.side == 'enemy' for s in enemy)
    assert all(s.source != 'color' for s in enemy if s.kind == 'building')
    assert scan(frame('base-reactors.png'), TACTICAL)              # nothing selected: the base is still there


def test_a_named_unit_keeps_its_name_after_its_selection_ends():
    tracker = Tracker(clock=lambda: 0.0)
    tracker.update([Seen((100, 100, 130, 130), 'ours', 'vehicle', 'bars', 'rhino_tank', True, 0.8, anchor=(115, 120))], 0.0)
    # Deselected: only its house-color patch is seen, a little further on.
    tracks = tracker.update([Seen((110, 114, 124, 128), 'ours', 'patch', 'color', anchor=(119, 122))], 1.0)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.name == 'rhino_tank' and track.kind == 'vehicle' and track.box == (104, 102, 134, 132)
    assert track.label(ITEMS, now=1.0) == 'Rhino 80%'
    assert track.label(ITEMS, now=4.0).endswith('3s')
    assert tracker.update([], 5.0) and not tracker.update([], 9.5)        # named: kept 8 s unseen, then gone


def test_tracks_follow_the_camera_and_a_jump_drops_them():
    canvas = frame('army-battle.png')
    camera = CameraShift()
    assert camera(canvas, TACTICAL) is None                                # nothing to compare yet
    delta = camera(moved(canvas, -40, 14), TACTICAL)
    assert abs(delta[0] + 40) <= 1 and abs(delta[1] - 14) <= 1
    assert camera(moved(canvas, -40, 14), TACTICAL) == (0, 0)
    assert camera(frame('enemy-base-siege.png'), TACTICAL) is None         # a jump: nothing in common
    tracker = Tracker(clock=lambda: 0.0)
    tracker.update([Seen((200, 200, 260, 250), 'ours', 'building', 'color')], 0.0)
    tracker.update([], 0.2, delta=(-40, 14))
    assert tracker.tracks[0].box == (160, 214, 220, 264)
    tracker.update([], 0.4, delta=(-500, 0))
    assert not tracker.tracks                                             # moved out of view
    tracker.update([Seen((200, 200, 260, 250), 'ours', 'building', 'color')], 0.6)
    assert not tracker.update([], 0.8, delta=None)                        # jumped


def test_scene_worker_scans_off_the_calling_thread_and_draws_fading_boxes():
    worker = SceneWorker(built=lambda: {'rhino_tank': 20, 'flak_track': 3, 'tesla_trooper': 8}, items=ITEMS, period=0.05)
    try:
        worker.submit(frame('army-selected.png'), TACTICAL)
        deadline = time.monotonic() + 10
        while worker.snapshot()[1]['scans'] < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        tracks, stats = worker.snapshot()
        assert stats['scans'] >= 1 and tracks and worker.error is None
        counts = summary(tracks, ITEMS)
        assert counts['ours'] and counts['tracked'] == len(tracks)
        now = time.monotonic()
        fresh = draw_scene(np.zeros((768, 1024, 3), np.uint8), tracks, ITEMS, now)
        stale = draw_scene(np.zeros((768, 1024, 3), np.uint8), tracks, ITEMS, now + 3.0)
        assert fresh.any() and fresh.sum() > stale.sum()                   # unseen boxes fade
        worker.clear()
        assert worker.snapshot()[0] == []
    finally:
        worker.close()
    assert not worker.thread.is_alive()
