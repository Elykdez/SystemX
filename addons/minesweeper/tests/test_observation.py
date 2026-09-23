import copy
import math

import pytest

from addons.minesweeper.observation import parse_observation
from addons.minesweeper.planner import candidates


def observation():
    return {
        "cells": [{"label": f"Row {r + 1}, column {c + 1}, closed",
                   "box": {"x": 100 + c * 24, "y": 100 + r * 24, "width": 24, "height": 24},
                   "hit": True} for r in range(9) for c in range(9)],
        "statusText": "Opened an area.",
        "difficultyText": "Beginner \u00b7 9\u00d79, 10 mines",
        "faceText": "\U0001f642",
        "viewport": {"width": 1024, "height": 768},
    }


def test_parses_visible_accessible_board():
    raw = observation()
    raw["cells"][0]["label"] = "Row 1, column 1, open empty square"
    raw["cells"][1]["label"] = "Row 1, column 2, 1 adjacent mine"
    raw["cells"][2]["label"] = "Row 1, column 3, 3 adjacent mines"
    raw["cells"][3]["label"] = "Row 1, column 4, flagged"
    board = parse_observation(raw)
    assert (board.rows, board.cols, board.mines_total, board.status) == (9, 9, 10, "playing")
    assert [cell.state for cell in board.cells[:5]] == ["0", "1", "3", "flag", "hidden"]
    assert board.cells[0].box.center == (112, 112)


@pytest.mark.parametrize("label", ["Row 1, column 1, unknown", "Row 0, column 1, closed",
                                  "Row 10, column 1, closed", "Row 1, column 1, 1 adjacent mines",
                                  "Row 1, column 1, 0 adjacent mines", "Row 1, column 1, 9 adjacent mines",
                                  "Row 1, column 1, 2 adjacent mine", "closed", None])
def test_rejects_unknown_or_invalid_labels(label):
    raw = observation()
    raw["cells"][0]["label"] = label
    with pytest.raises(ValueError):
        parse_observation(raw)


def test_rejects_missing_and_duplicate_cells():
    raw = observation()
    raw["cells"][1] = copy.deepcopy(raw["cells"][0])
    with pytest.raises(ValueError):
        parse_observation(raw)
    raw = observation()
    raw["cells"].pop()
    with pytest.raises(ValueError):
        parse_observation(raw)


def test_hit_metadata_is_retained_for_driver_not_all_board_gate():
    raw = observation()
    raw["cells"][0]["hit"] = False
    assert parse_observation(raw).cells[0].state == "hidden"
    assert raw["cells"][0]["hit"] is False
    raw["cells"][0]["hit"] = "true"
    with pytest.raises(ValueError):
        parse_observation(raw)


@pytest.mark.parametrize("size", [0, -1, math.nan, math.inf, True, "100"])
def test_rejects_invalid_viewport(size):
    raw = observation()
    raw["viewport"]["width"] = size
    with pytest.raises(ValueError):
        parse_observation(raw)


def test_rejects_invalid_geometry():
    raw = observation()
    raw["cells"][0]["box"]["width"] = 0
    with pytest.raises(ValueError):
        parse_observation(raw)


@pytest.mark.parametrize("text", ["Beginner", "Beginner \u00b7 8\u00d79, 10 mines",
                                 "Custom \u00b7 10000\u00d710000, 10 mines"])
def test_rejects_unrecognized_board_dimensions(text):
    raw = observation()
    raw["difficultyText"] = text
    with pytest.raises(ValueError):
        parse_observation(raw)


def test_observed_winning_face_requires_complete_board():
    raw = observation()
    raw["faceText"] = "\U0001f600"
    raw["statusText"] = ""
    with pytest.raises(ValueError, match="Winning face"):
        parse_observation(raw)
    for index, entry in enumerate(raw["cells"]):
        row, col = divmod(index, 9)
        description = "flagged" if index < 10 else "open empty square"
        entry["label"] = f"Row {row + 1}, column {col + 1}, {description}"
    assert parse_observation(raw).status == "won"


@pytest.mark.parametrize("face", ["", "unrecognized", "\U0001f60e"])
def test_rejects_unobserved_game_faces(face):
    raw = observation()
    raw["faceText"] = face
    with pytest.raises(ValueError, match="game face"):
        parse_observation(raw)


def test_observed_losing_face_and_revealed_mine_labels():
    raw = observation()
    raw["faceText"] = "\U0001f635"
    raw["statusText"] = "Boom! That square held a mine."
    raw["cells"][4]["label"] = "Row 1, column 5, the mine you hit"
    raw["cells"][18]["label"] = "Row 3, column 1, a hidden mine"
    board = parse_observation(raw)
    assert board.status == "lost"
    assert board.cells[4].state == board.cells[18].state == "mine"
    assert candidates(board) == []


@pytest.mark.parametrize("description", ["the mine you hit", "a hidden mine"])
def test_mine_labels_are_not_accepted_during_play(description):
    raw = observation()
    raw["cells"][0]["label"] = f"Row 1, column 1, {description}"
    with pytest.raises(ValueError, match="Unrecognized visible cell"):
        parse_observation(raw)


def test_inconsistent_losing_face_is_rejected():
    raw = observation()
    raw["faceText"] = "\U0001f635"
    with pytest.raises(ValueError, match="Losing face"):
        parse_observation(raw)
