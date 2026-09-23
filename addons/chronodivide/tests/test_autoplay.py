"""Autoplay's menu pilot on recorded menu screens."""

from pathlib import Path

import cv2

from addons.chronodivide.autoplay import MenuPilot
from addons.chronodivide.tools.menu import SCREENS

FIXTURES = Path(__file__).with_name('fixtures')


def test_setup_screen_gets_start_game_once_it_has_settled():
    pilot = MenuPilot()
    setup = cv2.imread(str(FIXTURES / 'frame-setup.png'))
    assert pilot.step(setup, None, 100.0) is None               # just appeared
    assert pilot.step(setup, None, 101.0) is None               # not settled yet
    assert pilot.step(setup, None, 102.5) == ('setup', SCREENS['setup'][1])
    assert pilot.step(setup, None, 104.0) is None               # presses are spaced out
    assert pilot.step(setup, None, 109.0) == ('setup', SCREENS['setup'][1])


def test_play_and_unknown_screens_are_never_clicked_nor_right_after_physical_input():
    pilot = MenuPilot()
    play = cv2.imread(str(FIXTURES / 'frame-play.png'))
    for t in (100.0, 103.0, 110.0):
        assert pilot.step(play, None, t) is None
    setup = cv2.imread(str(FIXTURES / 'frame-setup.png'))
    pilot.step(setup, None, 200.0)
    assert pilot.step(setup, 199.0, 203.0) is None              # someone used the mouse just now
    assert pilot.step(setup, 180.0, 203.0) == ('setup', SCREENS['setup'][1])


def test_donation_request_over_the_score_screen_gets_maybe_later():
    pilot = MenuPilot()
    donation = cv2.imread(str(FIXTURES / 'frame-donation.png'))
    pilot.step(donation, None, 10.0)
    assert pilot.step(donation, None, 12.5) == ('donation', SCREENS['donation'][1])


def test_main_menu_after_a_browser_start_gets_demo_mode_never_fullscreen():
    # The main menu's Fullscreen button sits where Continue is on the score screen, and its
    # header is close to "Skirmish Score": the closest template must win.
    pilot = MenuPilot()
    main = cv2.imread(str(FIXTURES / 'frame-main.png'))
    pilot.step(main, None, 10.0)
    assert pilot.step(main, None, 12.5) == ('main', SCREENS['main'][1])
    score = cv2.imread(str(FIXTURES / 'frame-score.png'))
    pilot.step(score, None, 30.0)
    assert pilot.step(score, None, 32.5) == ('score', SCREENS['score'][1])


def test_the_game_is_opened_in_the_browser_only_when_no_window_shows_it(monkeypatch):
    from types import SimpleNamespace
    from addons.chronodivide import window
    shown, opened = [], []
    game = SimpleNamespace(title='Red Alert 2: Chrono Divide - Google Chrome', hwnd=0)
    monkeypatch.setattr(window, 'windows', lambda: list(shown))

    def opener(url):
        opened.append(url)
        shown.append(game)
    assert window.ensure_game(opener=opener, sleep=lambda s: None) is game
    assert opened == [window.GAME_URL]
    assert window.ensure_game(opener=opener, sleep=lambda s: None) is game
    assert opened == [window.GAME_URL]                          # already open: not again


def test_a_pause_the_agent_caused_itself_is_resumed_after_a_moment():
    from types import SimpleNamespace
    from addons.chronodivide.unattended import Supervisor
    now = [100.0]
    armed = []
    player = SimpleNamespace(taken_over=lambda: False, paused_internally=lambda: 99.0, menu_open=lambda: False,
                             last_physical=lambda: None, arm_on_start=lambda: armed.append(now[0]),
                             wants_pointer=lambda: False, last_playing=(True, ''))
    supervisor = Supervisor(player, clock=lambda: now[0])
    assert supervisor.step() is None and not armed                  # 1 s after the pause: wait
    now[0] = 102.5
    assert supervisor.step() == 'resumed-internal' and armed == [102.5]
