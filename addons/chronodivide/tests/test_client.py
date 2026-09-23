"""Client profiles, layouts for any view size, and finding the game (window and view) like the
Minesweeper desktop player's calibration."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from desktop.target import DesktopPause, Geometry, Target
from addons.chronodivide.calibration import Setup, find_game, load_setup, pick_window, rank_windows, save_setup
from addons.chronodivide.client import ClientProfile
from addons.chronodivide.layout import Layout
from addons.chronodivide.perception import colors
from addons.chronodivide.perception.colors import HouseColor
from addons.chronodivide.perception.reader import Perception
from addons.chronodivide.recorder import GameView
from addons.chronodivide.world import World

FIXTURES = Path(__file__).with_name('fixtures')


def frame(name):
    return cv2.imread(str(FIXTURES / name))


def test_native_soviet_hud_is_recognized_before_mcv_deployment():
    profile = ClientProfile.load()
    canvas = frame('soviet-undeployed-native.png')
    assert find_game(canvas, profile).rect.tuple() == (0, 0, 1024, 768)
    world = World()
    perception = Perception(profile=profile, compressed=True)
    mode, readings, lane = perception.read(canvas, world)
    assert perception.modes.required == .7 and mode['mode'] == 'hud'
    assert world.credits.value == 10000 and world.sidebar_empty.value
    assert readings and lane is None and not world.structures


def test_native_idle_soviet_infantry_is_not_a_build_clock():
    from addons.chronodivide.perception.sidebar import read_slots
    profile = ClientProfile.load()
    canvas = np.zeros((768, 1024, 3), np.uint8)
    canvas[:, 850:] = frame('soviet-idle-infantry-native.png')
    catalog = profile.cameos()
    catalog.compressed = True
    slots = read_slots(canvas, profile.layout(), catalog)
    assert [(s.item, s.state) for s in slots[:4]] == [
        ('conscript', 'available'), ('engineer', 'available'),
        ('attack_dog', 'available'), ('tesla_trooper', 'available')]


def resized_view(canvas, width, height):
    """A frame as the game would draw it at another size: the sidebar at the right edge (its top part
    as it is, its bottom anchor at the new bottom), the tactical view cut to fit. Synthetic: only
    Chrono Divide's 1024x768 view has been recorded."""
    view = np.zeros((height, width, 3), np.uint8)
    left = width - 174
    view[:height - 32, :left] = canvas[:height - 32, :left]
    view[:height - 32, left:] = canvas[:height - 32, 850:]
    view[height - 32:, left:] = canvas[736:, 850:]
    view[height - 32:, :left] = canvas[736:, :left]
    return view


def on_page(canvas, client=(1822, 1329), at=(399, 337), page=64):
    image = np.full((client[1], client[0], 3), page, np.uint8)
    image[:90] = (40, 40, 40)                              # browser chrome
    image[at[1]:at[1] + canvas.shape[0], at[0]:at[0] + canvas.shape[1]] = canvas
    return image


@pytest.fixture(scope='module')
def cd():
    return ClientProfile.load('chronodivide')


def test_derived_layout_is_the_measured_one_at_1024x768(cd):
    derived, measured = cd.layout(1024, 768), Layout.for_canvas(1024, 768)
    for part in ('regions', 'tabs', 'points', 'anchors'):
        assert getattr(derived, part) == getattr(measured, part), part
    assert derived.grid == measured.grid and derived.menu == measured.menu


def test_derived_layout_follows_the_view_size(cd):
    small, wide = cd.layout(800, 600), cd.layout(1366, 768)
    assert small.tactical.w == 626 and small.tactical.h == 568 and small.grid.rows == 6
    assert small.regions['power'].h == 600 - 228 - 42 and small.points['park'] == (716, 522)
    assert wide.tactical.w == 1192 and wide.grid.rows == 10 and wide.regions['credits'].x == 1192 + 46


def test_starter_profile_for_the_original_game_loads(cd):
    original = ClientProfile.load('ra2-original')
    assert original.input == 'absolute' and not original.embedded and not original.verified
    assert 'gamemd.exe' in original.processes and cd.input == 'locked' and cd.embedded


def test_relative_external_profile_keeps_saved_templates_loadable(cd, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = cd.derived('Browser test')
    profile.path = Path('local/browser.json')
    catalog = cd.cameos()
    profile.save(catalog=catalog)
    restored = ClientProfile.load('local/browser.json').cameos()
    assert restored.templates.keys() == catalog.templates.keys()
    assert np.array_equal(restored.empty, catalog.empty)


def test_game_view_is_found_on_a_browser_page(cd):
    found = find_game(on_page(frame('frame-play.png')), cd)
    assert found.rect.tuple() == (399, 337, 1024, 768) and found.anchors['top_buttons'] >= 0.9


def test_game_view_is_found_in_an_app_window_and_with_a_side_panel(cd):
    client = np.full((887, 1094, 3), 60, np.uint8)
    client[:31] = (240, 240, 240)
    client[75:75 + 768, 35:35 + 1024] = frame('frame-play.png')
    assert find_game(client, cd).rect.tuple() == (35, 75, 1024, 768)
    # The skirmish setup in Chrome with a side panel: no in-match sidebar, so no anchors; once the
    # view's size is known (from a match, or the setup), the page locator finds it in menus too.
    panel = frame('client-chrome-sidepanel.png')
    assert find_game(panel, cd) is None
    view = GameView.__new__(GameView)
    view.profile, view.embedded, view.size, view.marked, view.known = cd, True, (1024, 768), None, None
    assert view.find(panel).tuple() == (127, 280, 1024, 768)


def test_game_filling_its_window_is_the_whole_client(cd):
    native = ClientProfile.load('ra2-original')
    assert find_game(frame('frame-play.png'), native).rect.tuple() == (0, 0, 1024, 768)


def test_game_menu_still_shows_where_the_view_is(cd):
    assert find_game(on_page(frame('frame-menu.png')), cd).rect.tuple() == (399, 337, 1024, 768)
    assert find_game(on_page(frame('frame-setup.png')), cd) is None        # no sidebar: not in a match


def test_a_view_of_another_size_is_found_and_read(cd):
    view = resized_view(frame('frame-play.png'), 800, 600)
    found = find_game(on_page(view, at=(311, 400)), cd)
    assert found.rect.tuple() == (311, 400, 800, 600)
    perception = Perception(profile=cd, layout=cd.layout(800, 600))
    world = World(clock=lambda: 1.0)
    mode, readings, lane = perception.read(view, world, 1.0)
    assert mode['mode'] == 'hud' and world.snapshot()['credits'] == 5200 and lane == 'buildings'


def fake_windows():
    return [Target(1, 10, 1, 'chronodivide'),                               # a File Explorer folder
            Target(2, 20, 2, 'MRStage - Google Chrome'),
            Target(3, 20, 2, 'Chrono Divide - Google Chrome'),
            Target(4, 30, 3, 'Red Alert 2')]


PROCESSES = {10: 'explorer.exe', 20: 'chrome.exe', 30: 'game.exe'}
CLASSES = {1: 'CabinetWClass'}


def rank(profile):
    return rank_windows(profile, fake_windows(), process=PROCESSES.get, klass=lambda hwnd: CLASSES.get(hwnd, 'X'))


def test_windows_are_ranked_by_title_and_process_and_folders_never_count(cd):
    ranked = rank(cd)
    assert [c.target.hwnd for c in ranked][:2] == [3, 2] and 1 not in [c.target.hwnd for c in ranked]
    original = rank(ClientProfile.load('ra2-original'))
    assert original[0].target.hwnd == 4 and original[0].titled
    kwargs = dict(process=PROCESSES.get, klass=lambda hwnd: CLASSES.get(hwnd, 'X'), foreground=lambda: 0)
    assert pick_window(cd, None, fake_windows(), **kwargs).hwnd == 3
    saved = Setup(window=dict(hwnd=2, pid=20, created=2))
    assert pick_window(cd, saved, fake_windows(), **kwargs).hwnd == 2         # the window picked in Setup
    with pytest.raises(DesktopPause):
        pick_window(cd, None, fake_windows()[:2], **kwargs)


def test_setup_is_saved_and_its_view_used_for_the_same_client_size(tmp_path):
    setup = Setup(profile='chronodivide')
    from addons.chronodivide.window import Rect
    setup.remember(Target(3, 20, 2, 'Chrono Divide'), 'chrome.exe', Rect(35, 75, 1024, 768), Geometry(0, 0, 1094, 887, 96))
    loaded = load_setup(save_setup(setup, tmp_path / 'setup.json'))
    assert loaded.size == (1024, 768) and loaded.window['process'] == 'chrome.exe'
    assert loaded.marked(Geometry(5, 5, 1094, 887, 96))[0].tuple() == (35, 75, 1024, 768)
    assert loaded.marked(Geometry(5, 5, 1200, 887, 96)) is None


def test_game_view_finds_a_marked_view_a_page_and_a_changed_size(cd):
    view = GameView.__new__(GameView)
    view.profile, view.embedded, view.size, view.marked, view.known = cd, True, (1024, 768), None, None
    assert view.find(on_page(frame('frame-play.png'))).tuple() == (399, 337, 1024, 768)
    assert view.find(on_page(resized_view(frame('frame-play.png'), 800, 600), at=(311, 400))).tuple() == (311, 400, 800, 600)
    assert view.size == (800, 600)
    from addons.chronodivide.window import Rect
    view.marked = (Rect(10, 20, 1024, 768), (1822, 1329))
    assert view.find(on_page(frame('frame-play.png'))).tuple() == (399, 337, 1024, 768)   # anchors first
    assert view.found_by == 'anchors'
    assert view.find(on_page(frame('frame-setup.png'))).tuple() == (10, 20, 1024, 768)   # no anchors: the setup's
    assert view.found_by == 'setup'


def test_clipped_canvas_blocks_capture_until_the_full_view_fits():
    from types import SimpleNamespace
    from addons.chronodivide.window import Rect
    geometry = Geometry(0, 0, 1200, 800, 96)
    frame = SimpleNamespace(pixels=np.zeros((800, 1200, 3), np.uint8), geometry=geometry)
    captures = []

    def capture(region=None, **kwargs):
        captures.append(region)
        return frame

    view = GameView.__new__(GameView)
    view.profile, view.embedded, view.size, view.known = None, True, (1024, 768), None
    view.marked = (Rect(0, 100, 1024, 768), (1200, 800))
    view.canvas = view.geometry = None
    view.on_found = view.relocate_seconds = None
    view.foreground, view.error = False, ''
    view.capture = SimpleNamespace(read=capture)
    view.target = SimpleNamespace(geometry=lambda: geometry, check_point=lambda *args: None)
    with pytest.raises(DesktopPause, match='canvas is clipped'):
        view.read()
    assert 'bottom bar' in view.error
    assert captures == [None]                 # never request a cropped, out-of-window frame
    view.marked = (Rect(0, 20, 1024, 768), (1200, 800))
    view.read()
    assert view.error == '' and captures[-1] == (0, 20, 1024, 768)


def test_old_fullscreen_size_does_not_hide_a_visible_windowed_menu(cd):
    view = GameView.__new__(GameView)
    view.profile, view.embedded, view.size = cd, True, (1404, 766)
    view.marked = view.known = None
    assert view.find(on_page(frame('frame-setup.png'))).tuple() == (399, 337, 1024, 768)
    assert view.size == (1024, 768)


def test_house_colors_measured_and_taught():
    canvas = frame('army-selected.png')
    assert HouseColor.from_json('red').mask(canvas).sum() > 1000
    patch = np.zeros((8, 8, 3), np.uint8)
    patch[:] = (40, 200, 40)                                               # a green house
    green = HouseColor.sample('green', patch)
    assert abs(green.hue - 60) <= 2 and green.to_json()['name'] == 'green'
    assert HouseColor.from_json(green.to_json()) == green
    with pytest.raises(ValueError):
        HouseColor.sample('none', np.full((8, 8, 3), 90, np.uint8))        # grey: no house color
    previous = colors.use(green, (HouseColor.from_json('red'),))
    try:
        from addons.chronodivide.perception.units import enemy_mask, house_mask
        assert enemy_mask(canvas).sum() > 1000 and house_mask(canvas).sum() < enemy_mask(canvas).sum()
    finally:
        colors.use(*previous)


def test_a_dark_page_gives_no_answer_and_the_view_found_in_the_match_holds(cd):
    """Live 2026-09-28: after the window was restored, the page around the canvas was dark and the
    page locator took the first of 2238 equally good places."""
    from addons.chronodivide.window import locate_canvas
    client = np.full((892, 1074, 3), 64, np.uint8)
    client[40:880, 10:1064] = 0                                        # a dark page around the canvas
    client[78:78 + 768, 25:25 + 1024] = frame('frame-setup.png')       # a menu: no sidebar anchors
    with pytest.raises(DesktopPause):
        locate_canvas(client)
    view = GameView.__new__(GameView)
    view.profile, view.embedded, view.size, view.marked, view.known = cd, True, (1024, 768), None, None
    view.found_by = None
    with pytest.raises(DesktopPause):
        view.find(client)
    playing = client.copy()
    playing[78:78 + 768, 25:25 + 1024] = frame('frame-play.png')
    assert view.find(playing).tuple() == (25, 78, 1024, 768)             # in a match: the anchors say
    assert view.find(client).tuple() == (25, 78, 1024, 768)              # back in a menu: the same place
