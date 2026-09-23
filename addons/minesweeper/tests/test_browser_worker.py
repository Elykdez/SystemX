"""Browser-inspector integration with fake pages and no OS input or model."""

from types import SimpleNamespace

import pytest

from addons.minesweeper.board import Board, Box, Cell
from addons.minesweeper.browser_driver import MinesweeperBrowser
from addons.minesweeper.session import StaleObservation


def board(states, status='playing'):
    return Board(1, len(states), tuple(Cell(0, col, state, Box(col * 20, 0, 20, 20))
                                      for col, state in enumerate(states)), status)


def test_pause_during_final_browser_hit_test_cancels_click():
    allowed = [True]
    clicks = []
    def hit_test(*args):
        allowed[0] = False
        return True
    page = SimpleNamespace(evaluate=hit_test, mouse=SimpleNamespace(click=lambda *a, **k: clicks.append(a)))
    driver = MinesweeperBrowser(page, can_act=lambda: allowed[0])
    with pytest.raises(StaleObservation, match='cancelled'):
        driver.click(board(('hidden',)).cells[0], 'left')
    assert not clicks


def test_other_cells_changing_does_not_confirm_selected_cell():
    observations = iter([board(('hidden', '0')), board(('0', '0'))])
    page = SimpleNamespace(wait_for_timeout=lambda _: None)
    driver = MinesweeperBrowser(page)
    driver.pending = (0, 0, 'left')
    driver.observe = lambda: next(observations)
    result = driver.wait_for_change(board(('hidden', 'hidden')))
    assert result.cells[0].state == '0'
    assert driver.pending is None


def test_browser_worker_uses_shared_session_and_honors_single_step(tmp_path, monkeypatch):
    from desktop.input import Gate
    from addons.minesweeper import browser_worker as worker
    gate = Gate()
    gate.start()
    events, clicks = [], []
    before = board(('0', 'hidden'))
    after = board(('0', '0'), 'won')
    current = [before]
    driver = SimpleNamespace(observe=lambda: current[0],
                             click=lambda cell, button: (clicks.append((cell.col, button)), current.__setitem__(0, after)),
                             wait_for_change=lambda _: after)
    monkeypatch.setattr(worker, 'MinesweeperBrowser', lambda page, can_act: driver)
    player = worker.BrowserWorker('http://test', 'beginner', 'http://model', gate,
                                  lambda kind, data: events.append((kind, data)))
    page = SimpleNamespace(screenshot=lambda: b'pixels')
    result = player.play_session(page, driver, tmp_path, gate.generation, True, 'logic', 5, .1)
    assert result['status'] == 'won'
    assert clicks == [(1, 'left')]
    assert [kind for kind, _ in events] == ['step', 'preview']
    assert (tmp_path / 'actions.jsonl').is_file()
    assert not gate.running


def test_cancelled_browser_generation_never_calls_driver(tmp_path, monkeypatch):
    from desktop.input import Gate
    from addons.minesweeper import browser_worker as worker
    gate = Gate()
    gate.start()
    generation = gate.generation
    gate.pause()
    driver = SimpleNamespace(observe=lambda: pytest.fail('Stale run observed the board'))
    monkeypatch.setattr(worker, 'MinesweeperBrowser', lambda page, can_act: driver)
    player = worker.BrowserWorker('http://test', 'beginner', 'http://model', gate, lambda *args: None)
    result = player.play_session(None, driver, tmp_path, generation, False, 'logic', 5, .1)
    assert result['status'] == 'paused' and result['steps'] == 0
