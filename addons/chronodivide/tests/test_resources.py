"""Native resource chips, recorded gem/ore fields and ambiguous enemy-base glints."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.perception.resources import gem_field, ore_supply, ore_target


FIXTURES = Path(__file__).with_name('fixtures')
BLUE, RED, GREEN = (190, 40, 30), (30, 40, 210), (30, 190, 40)


def test_browser_ore_supply_distinguishes_full_and_depleted_field_at_home_view_edge():
    counts = []
    for fixture in ('ore-rich-browser.png', 'ore-depleted-browser.png'):
        crop = cv2.imread(str(FIXTURES / fixture))
        assert crop is not None
        canvas = np.zeros((768, 1024, 3), np.uint8)
        canvas[600:736, 500:850] = crop
        counts.append(ore_supply(canvas, (425, 368), 900))
        assert ore_supply(canvas, (425, 368), 400) == 0
    assert counts[0] >= 12 and counts[1] < 12


def chips(canvas, x, y, cols=4, rows=3, color=BLUE):
    for row in range(rows):
        for col in range(cols):
            px, py = x + col * 13, y + row * 11
            canvas[py:py + 2, px:px + 3] = color


def mixed_gems(canvas, x, y):
    for row in range(8):
        for col in range(8):
            px, py = x + col * 10, y + row * 9
            canvas[py:py + 2, px:px + 3] = (BLUE, RED, GREEN)[(col + row) % 3]
            canvas[py + 2:py + 4, px + 2:px + 4] = (230, 230, 230)


def test_gem_field_returns_a_visible_chip_and_local_density():
    canvas = np.full((768, 1024, 3), (45, 80, 55), np.uint8)
    mixed_gems(canvas, 300, 380)
    mixed_gems(canvas, 900, 400)  # Sidebar is outside the tactical view.

    result = gem_field(canvas, (250, 350, 150, 110))

    assert result is not None
    (x, y), density = result
    assert 300 <= x <= 373 and 380 <= y <= 446
    assert tuple(canvas[y, x]) == BLUE
    assert density >= 58
    assert ore_target(canvas, (320, 390), 100) is None


def test_gem_field_abstains_on_isolated_chips_and_sidebar():
    canvas = np.zeros((768, 1024, 3), np.uint8)
    mixed_gems(canvas, 900, 400)
    chips(canvas, 400, 400, cols=2, rows=2)
    chips(canvas, 100, 100, cols=2, rows=2)
    chips(canvas, 700, 600, cols=2, rows=2)
    canvas[300:340, 300:340] = BLUE  # A large house-colored sprite is not a crystal field.
    assert gem_field(canvas) is None
    assert gem_field(None) is None


def test_gem_field_abstains_on_large_house_colored_sprite_with_glints():
    canvas = np.full((768, 1024, 3), (45, 80, 55), np.uint8)
    canvas[380:450, 300:380] = BLUE
    chips(canvas, 300, 380, rows=4, color=RED)
    chips(canvas, 345, 390, rows=4, color=GREEN)
    chips(canvas, 300, 430, rows=4, color=(230, 230, 230))
    assert gem_field(canvas) is None


@pytest.mark.parametrize('jpeg_quality', [None, 85])
def test_gem_field_recognizes_recorded_multicolor_crystals(jpeg_quality):
    image = cv2.imread(str(FIXTURES / 'gem-field-browser.png'))
    assert image is not None
    if jpeg_quality is not None:
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        assert ok
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    found = gem_field(image)
    assert found is not None and found[1] >= 58
    x, y = found[0]
    b, g, r = map(int, image[y, x])
    assert b - g >= 25 and b >= 90


@pytest.mark.parametrize('fixture', ['ore-field-browser.png', 'base-reactors.png',
                                     'reactor-glow-browser.png', 'frame-start.png',
                                     'gem-like-enemy-base-browser.png'])
def test_gem_field_abstains_on_real_non_gem_browser_frames(fixture):
    image = cv2.imread(str(FIXTURES / fixture))
    assert image is not None
    assert gem_field(image) is None
