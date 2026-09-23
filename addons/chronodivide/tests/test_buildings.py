"""Building appearance recognition on recordings not used to extract the templates.

yard-calibration-stall.png is frame 80 of local run 20260928-125050: the selected
construction yard during the pointer-calibration stall. Templates come only from
the older base-reactors.png recording; the health bar and Primary label are unseen.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.perception import colors
from addons.chronodivide.perception.buildings import BuildingRecognizer, default_recognizer

FIXTURES = Path(__file__).with_name('fixtures')
TACTICAL = (0, 0, 850, 736)


def frame(name):
    return cv2.imread(str(FIXTURES / name))


def test_selected_yard_in_failed_run_is_one_complete_building():
    found = default_recognizer().detect(frame('yard-calibration-stall.png'), TACTICAL)
    assert len(found) == 1
    yard = found[0]
    assert (yard.name, yard.side, yard.kind, yard.source) == (
        'construction_yard', 'ours', 'building', 'template')
    assert yard.box == (253, 327, 458, 504)
    assert yard.sure


def test_adjacent_yard_and_reactor_remain_two_buildings_on_heldout_frame():
    found = default_recognizer().detect(frame('enemy-near-base.png'), TACTICAL)
    assert [(b.name, b.box) for b in found] == [
        ('construction_yard', (325, 255, 530, 432)),
        ('tesla_reactor', (270, 350, 379, 450)),
    ]


def test_multiple_reactors_and_yard_in_heldout_populated_base():
    found = default_recognizer().detect(frame('army-selected.png'), TACTICAL)
    assert sum(b.name == 'construction_yard' for b in found) == 1
    reactors = [b for b in found if b.name == 'tesla_reactor']
    assert len(reactors) == 3
    assert {b.box[:2] for b in reactors} == {(120, 395), (690, 560), (570, 620)}
    assert all(not b.sure for b in reactors)  # animated/occluded appearances stay tentative


@pytest.mark.parametrize('name', ['frame-start.png', 'frame-play.png', 'frame-score.png',
                                  'army-battle.png'])
def test_units_placement_tiles_score_and_unknown_buildings_are_not_named(name):
    assert default_recognizer().detect(frame(name), TACTICAL) == []


def test_allied_yard_is_recognized_in_heldout_enemy_base():
    found = default_recognizer().detect(frame('enemy-base-siege.png'), TACTICAL)
    assert [(b.name, b.side, b.box) for b in found] == [
        ('construction_yard', 'enemy', (265, 0, 470, 162)),
        ('war_factory', 'enemy', (103, 72, 304, 224))]


def test_color_patches_without_sprite_detail_are_not_buildings():
    canvas = frame('yard-calibration-stall.png')
    patches = np.zeros_like(canvas)
    patches[colors.OWN.mask(canvas)] = (0, 0, 255)
    assert default_recognizer().detect(patches, TACTICAL) == []


def test_camera_translation_and_nonzero_region_use_canvas_coordinates():
    canvas = frame('yard-calibration-stall.png')
    shifted = cv2.warpAffine(canvas, np.float32([[1, 0, 37], [0, 1, -23]]), (1024, 768))
    found = default_recognizer().detect(shifted, (100, 100, 650, 550))
    assert len(found) == 1
    assert found[0].box == (290, 304, 495, 481)


def test_ownership_follows_profile_after_shape_recognition(monkeypatch):
    monkeypatch.setattr(colors, 'OWN', colors.BLUE)
    monkeypatch.setattr(colors, 'ENEMIES', (colors.RED,))
    found = default_recognizer().detect(frame('yard-calibration-stall.png'), TACTICAL)
    assert len(found) == 1 and found[0].side == 'enemy'


def test_cached_templates_and_small_view():
    assert default_recognizer() is default_recognizer()
    assert default_recognizer().detect(np.zeros((32, 32, 3), np.uint8), (0, 0, 32, 32)) == []


def test_damaged_yard_heldout_selected_frame_is_recognized_once():
    found = default_recognizer().detect(frame('repair-damaged-yard-browser.png'), TACTICAL)
    assert [(b.name, b.side, b.box) for b in found] == [
        ('construction_yard', 'ours', (325,255,530,432)),
        ('war_factory', 'ours', (88,369,269,524))]
    assert found[0].score > .95


@pytest.mark.parametrize('name', ['soviet-factory-healthy.png',
                                  'soviet-factory-damaged-selected.png'])
def test_factory_appearance_and_selected_damage_are_recognized_once(name):
    found = default_recognizer().detect(frame(name), (0, 0, 220, 220))
    assert [(b.name, b.side, b.box) for b in found] == [
        ('war_factory', 'ours', (18,59,199,214))]


def test_factory_ownership_uses_current_side_colors(monkeypatch):
    monkeypatch.setattr(colors, 'OWN', colors.BLUE)
    monkeypatch.setattr(colors, 'ENEMIES', (colors.RED,))
    found = default_recognizer().detect(frame('soviet-factory-healthy.png'), (0,0,220,220))
    assert len(found) == 1 and found[0].side == 'enemy'


@pytest.mark.parametrize('name', ['allied-factory-healthy.png',
                                  'allied-factory-healthy-selected.png',
                                  'allied-factory-damaged-unselected.png'])
def test_allied_factory_in_compressed_browser_capture(name, monkeypatch):
    monkeypatch.setattr(colors, 'OWN', colors.BLUE)
    monkeypatch.setattr(colors, 'ENEMIES', (colors.RED,))
    canvas = frame(name)
    found = default_recognizer().detect(canvas, compressed=True)
    assert [(b.name, b.side, b.box) for b in found] == [
        ('war_factory', 'ours', (13,47,214,199))]


def test_allied_factory_compressed_ownership_still_excludes_enemy(monkeypatch):
    canvas = frame('allied-factory-healthy.png')
    monkeypatch.setattr(colors, 'OWN', colors.RED)
    monkeypatch.setattr(colors, 'ENEMIES', (colors.BLUE,))
    found = default_recognizer().detect(canvas, compressed=True)
    assert len(found) == 1 and found[0].side == 'enemy'


@pytest.mark.parametrize('own,enemy,side', [(colors.BLUE, colors.RED, 'ours'),
                                          (colors.RED, colors.BLUE, 'enemy')])
def test_burning_allied_power_plant_is_recognized_in_heldout_capture(monkeypatch, own, enemy, side):
    monkeypatch.setattr(colors, 'OWN', own)
    monkeypatch.setattr(colors, 'ENEMIES', (enemy,))
    found = default_recognizer().detect(frame('allied-power-plant-damaged-heldout.png'), compressed=True)
    plants = [b for b in found if b.name == 'power_plant']
    assert len(plants) == 1
    assert plants[0].side == side and plants[0].box == (40, 55, 115, 140)
    assert plants[0].sure


def test_allied_factory_jpeg_colors_need_compressed_reader(monkeypatch):
    monkeypatch.setattr(colors, 'OWN', colors.BLUE)
    monkeypatch.setattr(colors, 'ENEMIES', (colors.RED,))
    assert default_recognizer().detect(frame('allied-factory-healthy.png')) == []


def test_template_variants_do_not_duplicate_one_building():
    recognizer = BuildingRecognizer()
    recognizer.templates.append(recognizer.templates[0])
    found = recognizer.detect(frame('repair-healthy-browser.png'), TACTICAL)
    assert sum(b.name == 'construction_yard' for b in found) == 1


@pytest.mark.parametrize('name', ['repair-healthy-browser.png', 'army-selected.png',
                                  'enemy-near-base.png', 'enemy-base-siege.png'])
def test_damaged_template_does_not_label_healthy_or_other_buildings(name):
    recognizer = BuildingRecognizer()
    recognizer.templates = [(row, template) for row, template in recognizer.templates
                            if row.get('template') == 'construction_yard_damaged']
    assert recognizer.detect(frame(name), TACTICAL) == []
