"""Visible Minesweeper observations, independent of browser and model code."""

from dataclasses import dataclass
import math
from typing import Literal


CellState = Literal["hidden", "flag", "mine", "0", "1", "2", "3", "4", "5", "6", "7", "8"]


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        for value in (self.x, self.y, self.width, self.height):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Box coordinates must be finite numbers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Box dimensions must be positive")

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    state: CellState
    box: Box

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.row, self.col)):
            raise ValueError("Cell coordinates must be nonnegative integers")
        if self.state not in ("hidden", "flag", "mine", *tuple(str(n) for n in range(9))):
            raise ValueError("Unknown visible cell state")
        if not isinstance(self.box, Box):
            raise ValueError("Cell requires a validated box")


@dataclass(frozen=True)
class Board:
    rows: int
    cols: int
    cells: tuple[Cell, ...]
    status: Literal["playing", "won", "lost"]
    mines_total: int | None = None

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.rows, self.cols)):
            raise ValueError("Board dimensions must be positive integers")
        if self.status not in ("playing", "won", "lost"):
            raise ValueError("Unknown board status")
        if self.mines_total is not None and (
            type(self.mines_total) is not int or not 0 <= self.mines_total <= self.rows * self.cols
        ):
            raise ValueError("Invalid total mine count")
        object.__setattr__(self, "cells", tuple(self.cells))
        positions = set()
        for cell in self.cells:
            if not isinstance(cell, Cell) or cell.row >= self.rows or cell.col >= self.cols:
                raise ValueError("Cell outside board")
            positions.add((cell.row, cell.col))
            if self.status == "playing" and cell.state == "mine":
                raise ValueError("A playing board cannot contain a revealed mine")
        if len(self.cells) != self.rows * self.cols or len(positions) != len(self.cells):
            raise ValueError("Board must cover every position exactly once")


@dataclass(frozen=True)
class Candidate:
    id: str
    row: int
    col: int
    button: Literal["left", "right"]
    reason: str
    guaranteed_safe: bool
    mine_probability: float | None
