from pathlib import Path

import cv2
import numpy as np

from addons.chronodivide.layout import Layout
from addons.chronodivide.perception.radar import distance, own_mask, read_radar


FIXTURES = Path(__file__).with_name('fixtures')
BOX = Layout.for_canvas(1024, 768).regions['radar']


def test_compressed_radar_finds_survivors_overlapping_enemy_dots():
    canvas = cv2.imread(str(FIXTURES / 'army-missing-radar-browser.png'))
    radar = read_radar(canvas, BOX, compressed=True)
    assert radar.online
    assert any(area >= 3 and distance(point, (969, 110)) <= 2 for area, point in radar.own)
    assert any(area >= 100 and distance(point, (908, 97)) <= 2 for area, point in radar.own)
    assert radar.enemy_near((972, 110), 5)


def test_compressed_radar_does_not_invent_survivors_after_actual_army_loss():
    canvas = np.zeros((768, 1024, 3), np.uint8)
    crop = cv2.imread(str(FIXTURES / 'radar-enemy-only-after-loss-browser.png'))
    canvas[BOX.y:BOX.y + BOX.h, BOX.x:BOX.x + BOX.w] = crop
    radar = read_radar(canvas, BOX, compressed=True)
    assert radar.online and radar.enemy_near((969, 108), 10)
    assert radar.own and all(distance(point, (908, 97)) <= 3 for _, point in radar.own)
    assert not any(distance(point, (969, 108)) < 20 for _, point in radar.own)


def test_native_red_mask_remains_strict_on_recorded_overlap():
    canvas = cv2.imread(str(FIXTURES / 'army-missing-radar-browser.png'))
    crop = BOX.crop(canvas)
    # Compare to the existing calibrated native house-color detector, not the JPEG path.
    from addons.chronodivide.perception import colors
    assert np.array_equal(own_mask(crop), colors.OWN.radar_mask(crop))
    overlap = crop[110 - BOX.y, 969 - BOX.x]
    assert int(overlap[2]) - int(overlap[0]) < 55


def test_allied_waypoint_lines_do_not_create_enemy_contacts_near_home():
    from addons.chronodivide.perception import colors
    previous = colors.use(colors.BLUE, (colors.RED,))
    try:
        for name in ('allied-radar-waypoint-native.png', 'allied-radar-diagonal-waypoint-native.png'):
            canvas = cv2.imread(str(FIXTURES / name))
            radar = read_radar(canvas, BOX, compressed=True, line_color='white')
            assert radar.online
            assert not radar.enemy_near((971, 111), 24), name
            assert any(area >= 100 and distance(point, (912, 97)) <= 2 for area, point in radar.enemy)
    finally:
        colors.use(*previous)
