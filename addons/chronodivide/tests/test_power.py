"""Power-bar regression from the completed speed-6 run 20260929-153705-008455.

power-yellow-browser.png: frame-1790667477454013200, healthy reactor + yard + refinery.
power-low-damaged-browser.png: frame-1790667647587672300, burning damaged reactor and
the visible game message 'Low Power - Production Slowed'. Both crop (856,228,14,498).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide import skills
from addons.chronodivide.layout import Layout
from addons.chronodivide.perception.hud import read_power
from addons.chronodivide.reflex import Reflex
from addons.chronodivide.world import World

FIXTURES = Path(__file__).with_name('fixtures')


def power(name):
    return read_power(cv2.imread(str(FIXTURES / name)))


def test_recorded_healthy_reactor_has_yellow_headroom_without_green():
    reading = power('power-yellow-browser.png')
    assert reading['level'] == 'ok'
    assert (reading['red_rows'], reading['yellow_rows'], reading['green_rows']) == (36, 44, 0)


def test_recorded_game_low_power_warning_matches_all_red_bar():
    reading = power('power-low-damaged-browser.png')
    assert reading['level'] == 'low'
    assert (reading['red_rows'], reading['yellow_rows'], reading['green_rows']) == (63, 0, 0)


@pytest.mark.parametrize('yellow_slice', [slice(30, 32), slice(80, 100)])
def test_thin_separator_or_yellow_below_consumption_is_not_spare_power(yellow_slice):
    crop = np.zeros((100, 14, 3), np.uint8)
    crop[40:80] = (0, 0, 220)
    crop[yellow_slice] = (0, 220, 220)
    assert read_power(crop)['level'] == 'low'


def test_corrected_power_allows_factory_but_real_shortage_still_prioritizes_reactor():
    world = World(clock=lambda: 100)
    world.structures.update(construction_yard=1, tesla_reactor=1, ore_refinery=1, barracks=1)
    world.set('credits', 5000)
    reflex = Reflex(skills.Context(Layout.for_canvas(1024, 768), world))
    world.set('power', power('power-yellow-browser.png'))
    assert world.power_margin() > 0
    assert reflex.next_building(['tesla_reactor', 'war_factory'], 100) == 'war_factory'
    world.set('power', power('power-low-damaged-browser.png'))
    assert reflex.next_building(['tesla_reactor', 'war_factory'], 100) == 'tesla_reactor'
