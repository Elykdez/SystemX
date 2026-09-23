"""Unit boxes: our selected units by their health bars (recorded game frames and drawn bars),
the enemy's objects by house color, names from what we built, and the inspector overlay."""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.perception.unitboxes import Unit, bar_frames, draw, enemy_objects, our_units, recognize
from addons.chronodivide.world import load_knowledge

FIXTURES = Path(__file__).with_name('fixtures')
ITEMS = load_knowledge()['items']


def frame(name):
    return cv2.imread(str(FIXTURES / name))


def draw_bar(canvas, x, y, length, health=1.0):
    """A selected unit's health bar as the game draws it: a white frame around two rows of pips,
    lit up to the unit's health, dark grey after."""
    white = (252, 252, 252)
    canvas[y, x:x + length] = canvas[y + 3, x:x + length] = white
    canvas[y + 1:y + 3, x] = canvas[y + 1:y + 3, x + length - 1] = white
    lit = x + 1 + round((length - 2) * health)
    for column in range(x + 1, x + length - 1):
        pip = (column - x) % 2 == 1
        if column < lit:
            canvas[y + 1, column] = (0, 252, 0) if pip else (0, 88, 0)
            canvas[y + 2, column] = (0, 196, 0) if pip else (0, 88, 0)
        else:
            canvas[y + 1:y + 3, column] = (0, 0, 0) if pip else (56, 56, 56)


def grass():
    return np.full((768, 1024, 3), (40, 120, 90), np.uint8)


def test_selected_units_are_boxed_under_their_bars_and_named_from_what_we_built():
    # Recorded game (2026-09-26): 13 Tesla Troopers, two Rhinos and two Flak Tracks selected (checked by eye).
    units = recognize(our_units(frame('army-selected.png')), {'rhino_tank': 20, 'flak_track': 3, 'tesla_trooper': 8},
                      ITEMS)
    assert Counter(u.kind for u in units) == {'infantry': 13, 'vehicle': 4}
    assert Counter(u.name for u in units) == {'tesla_trooper': 13, 'rhino_tank': 2, 'flak_track': 2}
    assert all(u.sure and u.side == 'ours' and u.health == 1.0 for u in units)
    for unit in units:
        x, y, length = unit.bar
        assert unit.box[1] > y and unit.box[0] < x + length and unit.box[2] > x        # the sprite below its bar


def test_a_badly_hurt_tank_is_still_a_vehicle():
    canvas = grass()
    draw_bar(canvas, 200, 300, 36, 0.15)                 # its lit pips are shorter than an infantry bar
    draw_bar(canvas, 300, 300, 18, 1.0)
    (tank, _), (trooper, _) = sorted(((u, u.box[0]) for u in our_units(canvas)), key=lambda p: p[1])
    assert tank.kind == 'vehicle' and tank.health < 0.3
    assert trooper.kind == 'infantry' and trooper.health == 1.0


def test_bars_side_by_side_or_overlapping_are_separate_units():
    canvas = grass()
    draw_bar(canvas, 100, 200, 18)
    draw_bar(canvas, 118, 200, 18)                       # touching end to end: not one vehicle bar
    draw_bar(canvas, 300, 200, 18)
    draw_bar(canvas, 309, 200, 18)                       # on top of the other, hiding its right end
    draw_bar(canvas, 500, 200, 36)
    draw_bar(canvas, 500, 203, 36)                       # stacked: one bar's bottom is the next one's top
    kinds = sorted(kind for *_, kind in bar_frames(canvas))
    assert kinds == ['infantry'] * 4 + ['vehicle'] * 2


def test_views_without_selected_units_have_no_unit_boxes():
    for name in ('frame-play.png', 'base-reactors.png', 'frame-start.png', 'enemy-near-base.png'):
        assert our_units(frame(name)) == [], name
    assert len(our_units(frame('frame-vehicles-queued.png'))) == 6        # six Conscripts selected there


def test_a_dense_battle_is_counted_and_fast():
    import time
    canvas = frame('army-battle.png')
    started = time.perf_counter()
    units = our_units(canvas)
    assert (time.perf_counter() - started) < 0.25                          # a few ms at 4 Hz for the inspector
    kinds = Counter(u.kind for u in units)
    assert 40 <= len(units) <= 60 and kinds['vehicle'] >= 20 and kinds['infantry'] >= 15


@pytest.mark.parametrize('quality', [80, 90, 95])
def test_compressed_screenshots_keep_selected_army_census(quality):
    for name, expected in [('army-selected.png', {'infantry': 13, 'vehicle': 4}),
                           ('frame-vehicles-queued.png', {'infantry': 6})]:
        encoded = cv2.imencode('.jpg', frame(name), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        canvas = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        units = our_units(canvas, compressed=True)
        assert Counter(u.kind for u in units) == expected
        assert all(u.health >= 0.9 for u in units)


def test_compressed_dense_battle_preserves_vehicle_count():
    encoded = cv2.imencode('.jpg', frame('army-battle.png'), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
    canvas = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    kinds = Counter(u.kind for u in our_units(canvas, compressed=True))
    assert kinds['vehicle'] == 29 and 20 <= kinds['infantry'] <= 23


def test_compressed_unselected_recordings_do_not_invent_units():
    for name in ('frame-play.png', 'base-reactors.png', 'frame-start.png', 'enemy-near-base.png'):
        encoded = cv2.imencode('.jpg', frame(name), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
        assert our_units(cv2.imdecode(encoded, cv2.IMREAD_COLOR), compressed=True) == [], name


def test_enemy_objects_are_boxed_as_units_and_buildings():
    enemy = enemy_objects(frame('enemy-base-siege.png'))
    kinds = Counter(u.kind for u in enemy)
    assert kinds['building'] >= 3 and all(u.side == 'enemy' for u in enemy)
    assert enemy_objects(frame('base-reactors.png')) == []


def test_names_are_guesses_when_several_types_fit():
    trooper = Unit((0, 0, 10, 20), 'ours', 'infantry', 1.0, (0, -5, 18))
    named, = recognize([trooper], {'conscript': 6, 'tesla_trooper': 2}, ITEMS)
    assert named.name == 'conscript' and not named.sure and named.label(ITEMS) == 'Conscript? 100%'
    flak = Unit((0, 0, 30, 20), 'ours', 'vehicle', 0.5, (0, -5, 36), cargo=True)
    named, = recognize([flak], {'rhino_tank': 9, 'flak_track': 1}, ITEMS)
    assert named.name == 'flak_track' and named.sure and named.label(ITEMS) == 'Flak 50%'


def test_the_overlay_draws_every_box_and_labels_where_there_is_room():
    canvas = grass()
    units = [Unit((100, 100, 140, 130), 'ours', 'vehicle', 0.8, name='rhino_tank', sure=True),
             Unit((102, 102, 142, 132), 'ours', 'vehicle', 0.6, name='rhino_tank', sure=True),
             Unit((400, 300, 420, 320), 'enemy', 'unit')]
    image = draw(canvas.copy(), units, ITEMS)
    assert (image[100, 100:140] != canvas[100, 100:140]).any()               # box edges drawn
    assert (image[102, 102:142] != canvas[102, 102:142]).any()
    assert (image[285:300, 400:430] != canvas[285:300, 400:430]).any()       # the enemy's label
