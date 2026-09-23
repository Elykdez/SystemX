import math

import pytest

from addons.minesweeper.board import Board, Box, Cell
from addons.minesweeper.planner import candidates


def board(*rows, status="playing", mines_total=None):
    names = {"?": "hidden", "F": "flag", "M": "mine"}
    cells = tuple(Cell(r, c, names.get(value, value), Box(c * 20, r * 20, 20, 20))
                  for r, row in enumerate(rows) for c, value in enumerate(row))
    return Board(len(rows), len(rows[0]), cells, status, mines_total)


def test_safe_reveal_from_existing_flag():
    moves = candidates(board("F1?"))
    assert [(m.row, m.col, m.button, m.mine_probability) for m in moves] == [(0, 2, "left", 0.0)]
    assert moves[0].guaranteed_safe


def test_proven_mine_is_flagged():
    move, = candidates(board("1?"))
    assert move.button == "right" and move.col == 1
    assert move.guaranteed_safe and move.mine_probability == 1.0


def test_subset_constraint_finds_safe_cell():
    # Left 1 constrains {a,b}; middle 1 constrains {a,b,c}, so c is safe.
    moves = candidates(board("11?", "???"))
    assert any(m.row == 1 and m.col == 2 and m.mine_probability == 0 for m in moves)
    assert all(m.guaranteed_safe for m in moves)


def test_subset_constraint_finds_mine():
    moves = candidates(board("13F", "???"))
    assert all(m.guaranteed_safe for m in moves)
    assert any(m.row == 1 and m.col == 2 and m.button == "right" for m in moves)


@pytest.mark.parametrize("rows", [("0F",), ("2?",), ("0?1",), ("03?", "???")])
def test_contradictory_observations_are_rejected(rows):
    with pytest.raises(ValueError, match="contradict|disagree|both"):
        candidates(board(*rows))


def test_unknown_opening_is_honest_and_bounded():
    moves = candidates(board("???", "???", "???"), 3)
    assert len(moves) == 3
    assert (moves[0].row, moves[0].col) == (1, 1)
    assert len({move.id for move in moves}) == 3
    assert all(not m.guaranteed_safe and m.mine_probability is None for m in moves)


def test_known_total_can_finish_remaining_cells():
    moves = candidates(board("F??", mines_total=1))
    assert len(moves) == 2
    assert all(m.mine_probability == 0 for m in moves)
    with pytest.raises(ValueError):
        candidates(board("FF?", mines_total=1))


@pytest.mark.parametrize("status", ["won", "lost"])
def test_terminal_board_has_no_actions(status):
    assert candidates(board("M?", status=status)) == []


def test_revealed_mine_is_not_a_playing_observation():
    with pytest.raises(ValueError):
        board("M?")


def test_box_center():
    assert Box(10, 20, 30, 40).center == (25, 40)


@pytest.mark.parametrize("values", [(0, 0, 0, 1), (0, 0, 1, -1), (math.nan, 0, 1, 1),
                                   (0, math.inf, 1, 1), (False, 0, 1, 1)])
def test_invalid_box(values):
    with pytest.raises(ValueError):
        Box(*values)


def test_board_requires_complete_unique_coverage():
    cell = Cell(0, 0, "hidden", Box(0, 0, 10, 10))
    for cells in [(cell,), (cell, cell)]:
        with pytest.raises(ValueError):
            Board(1, 2, cells, "playing")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        candidates(board("?"), limit)


def test_deductions_never_click_mines_across_small_valid_boards():
    # Exhaust every mine layout and every safe reveal subset on a 2x3 board.
    positions = [(r, c) for r in range(2) for c in range(3)]
    for mine_mask in range(64):
        mines = {p for index, p in enumerate(positions) if mine_mask & (1 << index)}
        safe = [p for p in positions if p not in mines]
        for reveal_mask in range(1 << len(safe)):
            revealed = {p for index, p in enumerate(safe) if reveal_mask & (1 << index)}
            rows = [["?"] * 3 for _ in range(2)]
            for r, c in revealed:
                rows[r][c] = str(sum(abs(r - mr) <= 1 and abs(c - mc) <= 1 for mr, mc in mines))
            observed = board(*( "".join(row) for row in rows))
            for move in candidates(observed):
                if move.guaranteed_safe:
                    assert ((move.row, move.col) in mines) == (move.button == "right")
