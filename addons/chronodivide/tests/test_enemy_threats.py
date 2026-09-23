"""Recorded deployed GIs must be distinguished from buildings and airborne infantry.

Positive crop: frame-1790650654097251300, [275:525,485:790], separate from the
template frame. Negative air crop: frame-1790650690997866300, [210:305,660:800].
Both are from run 20260929-105629-188247. These tests replay images without running a game.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.perception import colors
from addons.chronodivide.perception.enemy_threats import deployed_gi_targets

FIXTURES = Path(__file__).with_name('fixtures')


def frame(name):
    return cv2.imread(str(FIXTURES / name))


def detect(canvas, region=None):
    return deployed_gi_targets(canvas, region or (0, 0, canvas.shape[1], canvas.shape[0]), compressed=True)


def test_heldout_deployed_gis_are_seen_among_our_tanks_and_enemy_buildings():
    assert set(detect(frame('deployed-gi-heldout.png'))) == {(47, 130), (107, 162), (137, 27)}


def test_east_firing_gi_at_the_home_view_edge_remains_a_defense_target():
    from addons.chronodivide.army import raider_target
    from addons.chronodivide.layout import Box

    canvas = frame('deployed-gi-east-firing-browser.png')
    assert (69, 459) in detect(canvas, (0, 0, 850, 736))
    assert raider_target(canvas, Box(0, 0, 850, 736), (425, 366), 600,
                         compressed=True) == (69, 459)


@pytest.mark.parametrize('name', ['enemy-base-siege.png', 'base-reactors.png', 'army-selected.png',
                                  'army-battle.png', 'frame-start.png', 'rocketeers-heldout.png'])
def test_buildings_own_army_and_airborne_infantry_are_not_ground_gi_targets(name):
    canvas = frame(name)
    region = (0, 0, 850, 736) if canvas.shape[:2] == (768, 1024) else None
    assert detect(canvas, region) == []


def test_camera_translation_and_nonzero_region_return_canvas_points():
    source = frame('deployed-gi-heldout.png')
    canvas = np.zeros((300, 400, 3), np.uint8)
    canvas[20:270, 40:345] = source
    assert set(detect(canvas, (20, 10, 350, 280))) == {(87, 150), (147, 182), (177, 47)}


def test_enemy_color_without_body_and_sandbag_detail_is_insufficient():
    source = frame('deployed-gi-heldout.png')
    mask = colors.enemy_mask(source, compressed=True)
    canvas = np.zeros_like(source)
    canvas[mask] = source[mask]
    assert detect(canvas) == []


@pytest.mark.parametrize('name', ['deployed-gi-heldout.png', 'deployed-gi-east-firing-browser.png',
                                  'deployed-gi-allied-base-browser.png'])
def test_same_appearance_is_not_targeted_when_that_house_is_ours(monkeypatch, name):
    monkeypatch.setattr(colors, 'OWN', colors.BLUE)
    monkeypatch.setattr(colors, 'ENEMIES', (colors.RED,))
    assert detect(frame(name)) == []


def test_confirmed_defenders_take_priority_over_surrounding_buildings():
    from addons.chronodivide.army import battle_order, siege_target
    from addons.chronodivide.layout import Box
    canvas = frame('deployed-gi-heldout.png')
    region = Box(0, 0, canvas.shape[1], canvas.shape[0])
    assert battle_order(canvas, region, (160, 150), compressed=True) == ('fight', (107, 162))
    assert siege_target(canvas, region, (160, 150), compressed=True) is None


def test_allied_base_defender_takes_priority_over_siege_with_rocketeers_present():
    from addons.chronodivide.army import battle_order, siege_target
    from addons.chronodivide.layout import Box

    canvas = frame('deployed-gi-allied-base-browser.png')
    region = Box(0, 0, 850, 736)
    assert detect(canvas, (0, 0, 850, 736)) == [(503, 358)]
    assert battle_order(canvas, region, (545, 422), compressed=True) == ('fight', (503, 358))
    assert siege_target(canvas, region, (545, 422), compressed=True) is None
