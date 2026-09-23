"""Perception regression tests on frames recorded from the real game (Firefox, 1024x768)."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from desktop.target import Geometry
from addons.chronodivide.layout import Layout, PROFILES
from addons.chronodivide.perception.cursor import CursorFinder
from addons.chronodivide.perception.hud import CreditsReader, read_power
from addons.chronodivide.perception.modes import ModeReader, playing, pointer_locked
from addons.chronodivide.perception.reader import Perception
from addons.chronodivide.perception.sidebar import CameoCatalog, lane_of, read_slots
from addons.chronodivide.window import locate_canvas
from addons.chronodivide.world import World, load_knowledge

FIXTURES = Path(__file__).with_name('fixtures')
GEOMETRY = Geometry(31, 10, 1822, 1329, 96)


def frame(name):
    return cv2.imread(str(FIXTURES / f'frame-{name}.png'))


@pytest.fixture(scope='module')
def catalog():
    return CameoCatalog.load(PROFILES / 'cameos-1024x768.npz')


def test_v3_is_read_from_a_separate_browser_frame():
    canvas = cv2.imread(str(FIXTURES / 'v3-sidebar-browser.png'))
    catalog = CameoCatalog.load(PROFILES / 'cameos-1024x768.npz', compressed=True)
    slots = read_slots(canvas, Layout.for_canvas(1024, 768), catalog)
    assert slots[3].item == 'v3_launcher' and slots[3].state == 'available'


def test_iraq_desolator_and_shared_tesla_trooper_are_read_from_a_separate_browser_frame():
    canvas = cv2.imread(str(FIXTURES / 'iraq-infantry-browser.png'))
    catalog = CameoCatalog.load(PROFILES / 'cameos-1024x768.npz', compressed=True)
    slots = read_slots(canvas, Layout.for_canvas(1024, 768), catalog)
    assert (slots[6].item, slots[6].state) == ('desolator', 'available')
    assert (slots[4].item, slots[4].state) == ('tesla_trooper', 'available')
    assert lane_of(slots, load_knowledge()) == 'infantry'


def test_every_labeled_cameo_is_read_correctly_or_rejected(catalog):
    labels = json.loads((FIXTURES / 'cameos.json').read_text())
    with np.load(FIXTURES / 'cameos.npz') as crops:
        for key, (item, state) in labels.items():
            reading = catalog.classify(crops[key])
            assert (reading.item, reading.state) == (item, state), key


def test_build_progress_rises_with_the_clock(catalog):
    with np.load(FIXTURES / 'cameos.npz') as crops:
        progress = [catalog.classify(crops[k]).progress for k in ('1', '4', '7', '11')]
    assert progress == sorted(progress) and progress[0] < 0.15 and progress[-1] > 0.8


def test_sidebar_of_a_live_frame_identifies_the_visible_tab(catalog):
    layout = Layout.for_canvas(1024, 768)
    readings = read_slots(frame('play'), layout, catalog)
    known = [r for r in readings if r.item]
    assert known and all(r.state != 'unknown' for r in known)
    assert lane_of(readings, load_knowledge()) == 'buildings'
    assert sum(r.state == 'empty' for r in readings) >= 8


def test_basic_flak_infantry_cameo_is_available_in_browser_sidebar():
    sidebar = cv2.imread(str(FIXTURES / 'flak-trooper-browser.png'))
    canvas = np.zeros((768, 1024, 3), np.uint8)
    canvas[:, 850:] = sidebar
    catalog = CameoCatalog.load(PROFILES / 'cameos-1024x768.npz', compressed=True)
    readings = read_slots(canvas, Layout.for_canvas(1024, 768), catalog)
    assert (readings[2].item, readings[2].state) == ('flak_trooper', 'available')
    assert readings[3].item == 'attack_dog'
    assert lane_of(readings, load_knowledge()) == 'infantry'


def test_credits_digits_read_exactly():
    reader = CreditsReader.load(PROFILES / 'digits-1024x768.npz')
    with np.load(FIXTURES / 'credits.npz') as crops:
        for value in crops.files:
            assert reader.read(crops[value]) == int(value)
    assert reader.read(np.zeros((15, 96, 3), np.uint8)) is None


@pytest.mark.parametrize('quality', [70, 80, 90, 95])
def test_jpeg_credits_keep_digit_boundaries_without_relaxing_match_quality(quality):
    reader = CreditsReader.load(PROFILES / 'digits-1024x768.npz', compressed=True)
    with np.load(FIXTURES / 'credits.npz') as crops:
        for value in crops.files:
            # Browser screenshots are JPEG before the HUD is cropped, so the crop begins
            # partway through a compression block, just as it does in the real canvas.
            image = np.zeros((32, 112, 3), np.uint8)
            h, w = crops[value].shape[:2]
            image[3:3 + h, 8:8 + w] = crops[value]
            encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[3:3 + h, 8:8 + w]
            assert reader.read(decoded) == int(value), (quality, value)


def test_actual_browser_jpeg_credits_need_compressed_mode():
    # Captured IAB JPEG, cropped from canvas (896, 3, 96, 15); visually verified 10000.
    crop = cv2.imread(str(FIXTURES / 'credits-browser-10000.png'))
    native = CreditsReader.load(PROFILES / 'digits-1024x768.npz')
    compressed = CreditsReader.load(PROFILES / 'digits-1024x768.npz', compressed=True)
    assert native.read(crop) is None                 # native strict mode is unchanged
    assert compressed.read(crop) == 10000
    assert compressed.threshold == native.threshold == 0.12
    assert compressed.read(np.zeros_like(crop)) is None
    assert compressed.read(np.full_like(crop, 255)) is None


def test_jpeg_widening_does_not_exclude_the_correct_narrow_digit():
    image = cv2.imread(str(FIXTURES / 'credits-browser-wide-9830.png'))
    encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
    crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[3:18, 16:112]
    reader = CreditsReader.load(PROFILES / 'digits-1024x768.npz', compressed=True)
    assert reader.read(crop) == 9830                   # a widened 3 must not become an 8


def test_live_jpeg_525_splits_merged_digit_cores_without_relaxing_confidence():
    from addons.chronodivide.perception.hud import segment
    crop = cv2.imread(str(FIXTURES / 'credits-browser-525.png'))
    # Actual frame 1790650736385417100: the first two digits merged into a 10-column core.
    assert [width for _, width in segment(crop, threshold=215, compressed=True)] == [10, 5]
    reader = CreditsReader.load(PROFILES / 'digits-1024x768.npz', compressed=True)
    assert reader.read(crop) == 525
    assert reader.threshold == 0.12
    reader.templates['6'] = list(reader.templates['5'])
    reader._cache.clear()
    assert reader.read(crop) is None                  # splitting must not bypass tied-digit rejection


def test_compressed_credits_reject_ambiguous_glyphs():
    with np.load(FIXTURES / 'credits.npz') as crops:
        crop = crops['10000']
    reader = CreditsReader(compressed=True)
    assert reader.learn(crop, 10000)
    reader.templates['8'] = list(reader.templates['0'])
    assert reader.read(crop) is None                 # identical 0/8 evidence cannot invent a balance


def test_power_bar_reading_has_a_level():
    layout = Layout.for_canvas(1024, 768)
    power = read_power(layout.regions['power'].crop(frame('play')))
    assert power['level'] in ('ok', 'low') and 0 < power['fill'] <= 1


def test_modes_distinguish_play_menu_and_setup():
    reader = ModeReader.load(PROFILES / 'hud-1024x768.npz')
    assert reader.read(frame('play'))['mode'] == 'hud'
    assert reader.read(frame('menu'))['mode'] == 'menu'
    assert reader.read(frame('setup'))['mode'] == 'other'


@pytest.mark.parametrize('changed_share,expected', [(.3, 'hud'), (.31, 'other')])
def test_hud_matching_requires_seventy_percent(changed_share, expected):
    canvas = frame('play').copy()
    reader = ModeReader.from_frame(canvas)
    x, y, w, h = reader.anchors['sidebar_bottom']
    crop = canvas[y:y + h, x:x + w].copy()
    pixels = crop.reshape(-1, 3)
    count = int(len(pixels) * changed_share)
    pixels[:count] = (pixels[:count].astype(np.int16) + 127) % 256
    canvas[y:y + h, x:x + w] = crop
    assert reader.required == .7 and reader.read(canvas)['mode'] == expected


@pytest.mark.parametrize('fixture,intact', [
    ('browser-mode-overlay.npz', 'top_buttons_upper'),
    ('browser-mode-left-overlay.npz', 'top_buttons_right'),
])
def test_compressed_hud_accepts_unobscured_buttons_above_radar_overlay(fixture, intact):
    # Actual rejected browser frame (2026-09-28): retain the reader's three observed regions
    # and its taught templates, without storing the unrelated tactical screenshot.
    from addons.chronodivide.perception.modes import ANCHORS, MENU_REGION
    canvas = np.zeros((768, 1024, 3), np.uint8)
    with np.load(FIXTURES / fixture) as data:
        for name, box in {**ANCHORS, 'menu': MENU_REGION}.items():
            x, y, w, h = box
            canvas[y:y + h, x:x + w] = data[name]
        templates = {name: data['template:' + name] for name in ANCHORS}
        native = ModeReader(templates, required=.9)
        browser = ModeReader(templates, compressed=True, required=.9)
    assert native.read(canvas)['mode'] == 'menu'
    assert browser.read(canvas)['mode'] == 'hud'
    assert browser.read(canvas)['scores'][intact] >= browser.required
    x, y, w, h = MENU_REGION
    canvas[y:y + h, x:x + w] = frame('menu')[y:y + h, x:x + w]
    assert browser.read(canvas)['mode'] == 'menu'  # matching top buttons cannot override menu evidence


def test_compressed_hud_accepts_the_footer_beside_a_building_tooltip():
    from addons.chronodivide.perception.modes import ANCHORS, MENU_REGION
    canvas = np.zeros((768, 1024, 3), np.uint8)
    with np.load(FIXTURES / 'browser-mode-tooltip.npz') as data:
        for name, (x, y, w, h) in {**ANCHORS, 'menu': MENU_REGION}.items():
            canvas[y:y + h, x:x + w] = data[name]
        templates = {name: data['template:' + name] for name in ANCHORS}
    assert ModeReader(templates, required=.9).read(canvas)['mode'] == 'other'
    reader = ModeReader(templates, compressed=True, required=.9)
    result = reader.read(canvas)
    assert result['scores']['sidebar_bottom'] < reader.required
    assert result['mode'] == 'hud'
    assert result['scores']['sidebar_bottom_right'] >= reader.required
    x, y, w, h = MENU_REGION
    canvas[y:y + h, x:x + w] = frame('menu')[y:y + h, x:x + w]
    assert reader.read(canvas)['mode'] == 'menu'


def test_compressed_mode_still_blocks_pause_and_non_game_screens():
    def jpeg(canvas):
        encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    reader = ModeReader.from_frame(jpeg(frame('play')), compressed=True)
    assert reader.read(jpeg(frame('play')))['mode'] == 'hud'
    assert reader.read(jpeg(frame('menu')))['mode'] == 'menu'
    for name in ('setup', 'main', 'score'):
        assert reader.read(jpeg(frame(name)))['mode'] == 'other'


def test_agent_only_plays_when_the_match_is_actually_being_played():
    reader = ModeReader.load(PROFILES / 'hud-1024x768.npz')
    locked = dict(clipped=True, clip=[36, 15, 1848, 1334])
    free = dict(clipped=False, clip=[0, 0, 2560, 1440])
    hud, menu = reader.read(frame('play')), reader.read(frame('menu'))
    assert playing(hud, locked, GEOMETRY, True) == (True, 'playing')
    assert not playing(menu, locked, GEOMETRY, True)[0]             # Esc menu = paused
    assert not playing(hud, free, GEOMETRY, True)[0]                # pointer released
    assert not playing(hud, locked, GEOMETRY, False)[0]             # another window in front
    assert not playing(reader.read(frame('setup')), locked, GEOMETRY, True)[0]
    assert pointer_locked(locked, GEOMETRY) and not pointer_locked(free, GEOMETRY)


def test_cursor_sprite_is_found_at_its_hotspot():
    finder = CursorFinder.load()
    with np.load(FIXTURES / 'cameos.npz') as crops:
        slot = crops['17']                       # arrow cursor over an empty cameo slot
    canvas = frame('start').copy()
    canvas[228 + 50:228 + 50 + 48, 877:877 + 60] = slot
    assert finder.find(canvas, (850, 180, 174, 588)) == (877 + 13, 278 + 19)


def test_canvas_is_located_inside_the_browser_client_area():
    client = np.full((1329, 1822, 3), 64, np.uint8)
    client[:90] = (40, 40, 40)                  # browser chrome
    client[337:337 + 768, 399:399 + 1024] = frame('play')
    rect = locate_canvas(client)
    assert (rect.x, rect.y, rect.width, rect.height) == (399, 337, 1024, 768)


def test_mostly_black_canvas_with_a_narrow_margin_is_located():
    client = np.full((887, 1094, 3), 60, np.uint8)       # app window: title bar, gray page
    client[:31] = (240, 240, 240)
    client[75:75 + 768, 35:35 + 1024] = frame('setup')   # menu screen, mostly black
    rect = locate_canvas(client)
    assert (rect.x, rect.y, rect.width, rect.height) == (35, 75, 1024, 768)


def test_queued_units_show_counts_and_keep_the_lane_busy():
    world = World(clock=lambda: 10.0)
    mode, readings, lane = Perception().read(frame('vehicles-queued'), world, 10.0)
    queued = {r.item for r in readings if r.queued}
    assert lane == 'vehicles' and queued == {'war_miner', 'flak_track', 'terror_drone'}
    assert world.snapshot()['lanes']['vehicles']['status'] == 'building'    # no new queue click


def test_compressed_queue_counts_keep_all_three_productions_visible():
    canvas = frame('vehicles-queued')
    encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
    catalog = CameoCatalog.load(PROFILES / 'cameos-1024x768.npz', compressed=True)
    readings = read_slots(cv2.imdecode(encoded, cv2.IMREAD_COLOR), Layout.for_canvas(1024, 768), catalog)
    assert {r.item for r in readings if r.queued} == {'war_miner', 'flak_track', 'terror_drone'}


def test_compressed_cameo_art_keeps_labeled_states_without_false_queue_counts():
    catalog = CameoCatalog.load(PROFILES / 'cameos-1024x768.npz', compressed=True)
    labels = json.loads((FIXTURES / 'cameos.json').read_text())
    with np.load(FIXTURES / 'cameos.npz') as crops:
        for key, (item, state) in labels.items():
            encoded = cv2.imencode('.jpg', crops[key], [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
            reading = catalog.classify(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
            assert (reading.item, reading.state) == (item, state), key
    canvas = frame('play')
    encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
    readings = read_slots(cv2.imdecode(encoded, cv2.IMREAD_COLOR), Layout.for_canvas(1024, 768), catalog)
    assert not any(r.queued for r in readings)


def test_browser_reactor_ready_text_survives_jpeg_color_bleeding():
    # Live isolated browser, 2026-09-28: JPEG screenshot of a completed reactor.
    # This is the decoded first cameo crop, saved losslessly to avoid recompression.
    crop = cv2.imread(str(FIXTURES / 'reactor-ready-browser.png'))
    catalog = CameoCatalog.load(PROFILES / 'cameos-1024x768.npz', compressed=True)
    reading = catalog.classify(crop)
    assert (reading.item, reading.state, reading.progress) == ('tesla_reactor', 'ready', 1.0)
    yellow_patch = crop.copy()
    yellow_patch[4:22, 12:48] = (100, 230, 240)
    assert not catalog.is_ready(yellow_patch)               # yellow alone is not the word Ready


def test_perception_updates_the_world_from_one_frame():
    world = World(clock=lambda: 10.0)
    mode, readings, lane = Perception().read(frame('play'), world, 10.0)
    snapshot = world.snapshot()
    assert mode['mode'] == 'hud' and lane == 'buildings'
    assert snapshot['credits'] == 5200 and snapshot['tab'] == 'buildings'
    assert snapshot['lanes']['buildings']['available']


def test_mcv_and_base_are_found_by_house_color():
    from addons.chronodivide.perception.units import find_base, find_mcv
    start, play = frame('start'), frame('play')
    mcv = find_mcv(start)
    assert mcv is not None and abs(mcv[0] - 430) <= 8 and abs(mcv[1] - 358) <= 8
    assert find_base(start) is None                   # nothing deployed yet
    assert find_mcv(play) is None                     # a base is visible: never treat it as an MCV
    base = find_base(play)
    assert base is not None and 300 <= base[0] <= 550 and 200 <= base[1] <= 400
    assert find_mcv(frame('setup')) is None


def test_empty_power_bar_is_not_low_power():
    empty = np.zeros((498, 14, 3), np.uint8)
    empty[490] = (0, 0, 200)                          # the single red row seen at match start
    assert read_power(empty)['level'] == 'none'
    low = np.zeros((498, 14, 3), np.uint8)
    low[440:480] = (0, 0, 200)
    assert read_power(low)['level'] == 'low'
    ok = low.copy()
    ok[300:440] = (0, 200, 0)
    assert read_power(ok)['level'] == 'ok'


def test_canvas_is_located_with_a_darker_browser_side_panel_open():
    # Chrome with the Claude side panel: the window's right edge is the panel, not the page.
    client = cv2.imread(str(FIXTURES / 'client-chrome-sidepanel.png'))
    rect = locate_canvas(client)
    assert (rect.x, rect.y, rect.width, rect.height) == (127, 280, 1024, 768)


def test_power_surplus_can_be_green_or_yellow():
    # Live game 2026-09-25: a busy base draws much red under the green surplus; that is fine.
    surplus = cv2.imread(str(FIXTURES / 'power-surplus.png'))
    short = cv2.imread(str(FIXTURES / 'power-short.png'))
    assert read_power(surplus)['level'] == 'ok' and read_power(surplus)['red_rows'] > read_power(surplus)['green_rows']
    # This historical filename was mislabeled: its substantial yellow surplus is not low power.
    assert read_power(short)['level'] == 'ok'


def test_cursor_found_below_its_expected_spot_despite_the_sprite_hanging_down():
    # Live game 8: believed at the park spot (940, 103), the arrow really was at (936, 133).
    from addons.chronodivide.perception.cursor import CursorFinder
    from addons.chronodivide.skills import Context
    ctx = Context(Layout.for_canvas(1024, 768), World(), CursorFinder.load())
    canvas = cv2.imread(str(FIXTURES / 'cursor-drifted.png'))
    assert ctx.find_cursor(canvas, (940, 103), radius=40) == (936, 133)


def test_the_small_radar_cursor_is_found_where_the_big_arrow_is_not():
    # Live game 17: once the radar came online, the cursor parked over it was drawn as a small
    # white arrow, the finder lost it, and every recalibration failed.
    from addons.chronodivide.perception.cursor import CursorFinder
    finder = CursorFinder.load()
    radar = cv2.imread(str(FIXTURES / 'cursor-radar.png'))
    assert finder.find(radar, (850, 0, 174, 768)) == (941, 105)
    assert finder.find(radar, (900, 70, 100, 80)) == (941, 105)
