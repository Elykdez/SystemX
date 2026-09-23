"""Two supported GI poses recover the held-out speed-6 raid without weakening house masks."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.perception import colors
from addons.chronodivide.perception.enemy_threats import standing_gi_targets

FIXTURES = Path(__file__).with_name('fixtures')


def frame(name):
    return cv2.imread(str(FIXTURES / name))


def detect(canvas, region=None):
    region = region or (0, 0, min(850, canvas.shape[1]), min(736, canvas.shape[0]))
    return standing_gi_targets(canvas, region, compressed=True)


def test_all_three_attackers_in_heldout_selected_refinery_frame_are_recognized():
    canvas = frame('repair-damaged-refinery-browser.png')
    assert set(detect(canvas)) == {(742, 463), (708, 462), (708, 492)}
    # Compression erased every strict house-color pixel in this attacker's body. Its matched
    # full-color uniform and rifle/body pose supply evidence that color components cannot.
    assert not colors.enemy_mask(canvas[450:475, 699:718], compressed=True).any()


@pytest.mark.parametrize('name', ['base-reactors.png', 'reactor-glow-browser.png', 'army-selected.png',
                                  'army-battle.png', 'enemy-base-siege.png', 'frame-start.png',
                                  'rocketeers-heldout.png', 'deployed-gi-heldout.png'])
def test_reactors_own_units_terrain_and_other_enemy_poses_are_not_standing_gis(name):
    assert detect(frame(name)) == []


def test_nonzero_region_and_camera_translation_return_fresh_canvas_points():
    canvas = frame('repair-damaged-refinery-browser.png')
    moved = cv2.warpAffine(canvas, np.float32([[1, 0, -31], [0, 1, 17]]), (1024, 768))
    assert set(detect(moved, (650, 400, 170, 150))) == {(711, 480), (677, 479), (677, 509)}


def test_uniform_color_is_required_not_just_a_dark_infantry_shape():
    canvas = frame('repair-damaged-refinery-browser.png')
    gray = cv2.cvtColor(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert detect(gray) == []


def test_blue_house_fragments_without_sprite_appearance_do_not_confirm_a_raid():
    canvas = frame('repair-damaged-refinery-browser.png')
    fragments = np.zeros_like(canvas)
    mask = colors.enemy_mask(canvas, compressed=True)
    fragments[mask] = canvas[mask]
    assert detect(fragments) == []


def test_blue_template_is_disabled_when_blue_is_our_house(monkeypatch):
    monkeypatch.setattr(colors, 'OWN', colors.BLUE)
    monkeypatch.setattr(colors, 'ENEMIES', (colors.RED,))
    assert detect(frame('repair-damaged-refinery-browser.png')) == []


def test_small_empty_region_abstains():
    assert detect(np.zeros((10, 10, 3), np.uint8)) == []


def test_full_color_matching_is_limited_to_candidate_sprite_patches(monkeypatch):
    match = cv2.matchTemplate
    color_searches = []

    def bounded_match(image, template, method):
        if image.ndim == 3:
            color_searches.append(image.shape)
            assert image.shape == template.shape
        return match(image, template, method)

    monkeypatch.setattr(cv2, 'matchTemplate', bounded_match)
    assert set(detect(frame('repair-damaged-refinery-browser.png'))) == {(742, 463), (708, 462), (708, 492)}
    assert color_searches
