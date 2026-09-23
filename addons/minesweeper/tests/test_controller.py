from types import SimpleNamespace

import pytest

from addons.minesweeper.board import Board, Box, Cell
from addons.minesweeper.browser_driver import MinesweeperBrowser
from addons.minesweeper.session import MinesweeperSession, StaleObservation
from addons.minesweeper.__main__ import Controls


def board(offset=0, state="hidden"):
    return Board(1, 1, (Cell(0, 0, state, Box(offset, 0, 20, 20)),), "playing")


def test_pause_during_selection_prevents_input():
    clicks = []
    driver = SimpleNamespace(observe=lambda: board(), click=lambda *a: clicks.append(a))
    session = MinesweeperSession(driver, can_act=lambda: False)
    assert session.step()["status"] == "paused"
    assert not clicks


def test_console_pause_cancels_single_step_and_quit_wins_over_resume():
    controls = Controls()
    controls.commands.put("s")
    assert controls.poll()
    controls.commands.put("p")
    assert not controls.poll()
    controls.commands.put("q")
    controls.commands.put("r")
    assert not controls.poll()


@pytest.mark.parametrize("hit", [False, True])
def test_input_is_sent_only_after_successful_hit_test(hit):
    clicks = []
    page = SimpleNamespace(evaluate=lambda *a: hit,
                           mouse=SimpleNamespace(click=lambda *a, **kw: clicks.append((a, kw))))
    driver = MinesweeperBrowser(page)
    if hit:
        driver.click(board().cells[0], "right")
        assert clicks == [((10.0, 10.0), {"button": "right"})]
    else:
        with pytest.raises(StaleObservation):
            driver.click(board().cells[0], "left")
        assert not clicks


def test_geometry_change_is_not_treated_as_successful_click():
    driver = MinesweeperBrowser(None)
    driver.observe = lambda: board(offset=3)
    with pytest.raises(StaleObservation, match="geometry"):
        driver.wait_for_change(board())


def test_no_click_result_times_out_without_retrying_input():
    page = SimpleNamespace(wait_for_timeout=lambda ms: None)
    driver = MinesweeperBrowser(page, action_timeout=0)
    with pytest.raises(TimeoutError):
        driver.wait_for_change(board())
