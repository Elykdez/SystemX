"""Strict adapter for the Minesweeper page's rendered accessibility labels."""

import math
from pathlib import Path
import re

from .board import Board, Box, Cell


OBSERVE_SCRIPT = Path(__file__).with_name("observe.js").read_text(encoding="utf-8")
_LABEL = re.compile(r"Row ([1-9][0-9]*), column ([1-9][0-9]*), (.+)")
_NUMBER = re.compile(r"([1-8]) adjacent (mine|mines)")
_DIFFICULTY = re.compile(r"(Beginner|Intermediate|Expert) \u00b7 ([1-9][0-9]*)\u00d7([1-9][0-9]*), ([1-9][0-9]*) mines")
_SIZES = {"Beginner": (9, 9, 10), "Intermediate": (16, 16, 40), "Expert": (30, 16, 99)}


def _text(raw, key):
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Observation {key} must be text")
    return value


def _state(description, lost=False):
    if description == "closed":
        return "hidden"
    if description == "flagged":
        return "flag"
    if description == "open empty square":
        return "0"
    if lost and description in ("the mine you hit", "a hidden mine"):
        return "mine"
    number = _NUMBER.fullmatch(description)
    if number and (number[2] == "mine") == (number[1] == "1"):
        return number[1]
    raise ValueError(f"Unrecognized visible cell description: {description!r}")


def parse_observation(raw: dict) -> Board:
    """Validate a DOM observation without reading concealed game data.

    Hit-test metadata stays in the raw observation for the driver to check again
    immediately before clicking. An unrelated offscreen cell need not block a move.
    """
    if not isinstance(raw, dict):
        raise ValueError("Observation must be an object")
    viewport = raw.get("viewport")
    if not isinstance(viewport, dict):
        raise ValueError("Observation requires viewport dimensions")
    for key in ("width", "height"):
        value = viewport.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Viewport dimensions must be finite positive numbers")

    difficulty = _DIFFICULTY.fullmatch(_text(raw, "difficultyText"))
    if not difficulty:
        raise ValueError("Unrecognized visible difficulty")
    cols, rows, mines = (int(difficulty[i]) for i in (2, 3, 4))
    if (cols, rows, mines) != _SIZES[difficulty[1]]:
        raise ValueError("Visible difficulty dimensions do not match its name")
    if rows > 32 or cols > 64 or rows * cols > 1024:
        raise ValueError("Observed board exceeds supported dimensions")
    _text(raw, "statusText")
    face = _text(raw, "faceText")
    entries = raw.get("cells")
    if not isinstance(entries, list) or len(entries) != rows * cols:
        raise ValueError("Observation must contain every board cell")
    cells = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Cell observation must be an object")
        label = _LABEL.fullmatch(_text(entry, "label"))
        if not label:
            raise ValueError("Unrecognized visible cell label")
        if type(entry.get("hit")) is not bool:
            raise ValueError("Cell observation requires boolean hit-test metadata")
        box = entry.get("box")
        if not isinstance(box, dict) or set(box) != {"x", "y", "width", "height"}:
            raise ValueError("Cell observation requires a bounding box")
        cells.append(Cell(int(label[1]) - 1, int(label[2]) - 1,
                          _state(label[3], lost=face == "\U0001f635"), Box(**box)))
    status = "playing"
    if face == "\U0001f600":
        flags = sum(cell.state == "flag" for cell in cells)
        opened = sum(cell.state.isdigit() for cell in cells)
        if flags != mines or opened != rows * cols - mines:
            raise ValueError("Winning face disagrees with visible board completion")
        status = "won"
    elif face == "\U0001f635":
        if not any(cell.state == "mine" for cell in cells):
            raise ValueError("Losing face disagrees with visible board")
        status = "lost"
    elif face != "\U0001f642":
        raise ValueError("Unrecognized visible game face")
    return Board(rows, cols, tuple(cells), status, mines)
