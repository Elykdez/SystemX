"""Placement ghost: in placement mode the game draws the building's footprint under the cursor,
green where a cell is free and red where it is blocked (buildings, units, cliffs, too far).

Measured on Chrono Divide's 1024x768 view: ghost green is about BGR (23, 220, 51) and ghost
red about (17, 38, 213). That red is also our own house color, so the ghost is measured
against a reference frame taken without it: only pixels that changed and have a ghost color
count. The footprint surrounds the cursor (a War Factory spans about 300x120 pixels).
"""

from dataclasses import dataclass

import numpy as np

from . import view

GREEN = np.array([23, 220, 51], np.int16)        # BGR
RED = np.array([17, 38, 213], np.int16)
TOLERANCE = np.array([25, 25, 30], np.int16)
CHANGED = 40                                     # min channel difference from the reference
WINDOW = (240, 160)                              # half width and height searched around the cursor


@dataclass(frozen=True)
class Ghost:
    green: int                                   # ghost-green pixels (free cells)
    red: int                                     # ghost-red pixels (blocked cells)
    green_center: tuple | None = None
    red_center: tuple | None = None

    @property
    def seen(self):
        """A footprint is drawn near the cursor (we are in placement mode)."""
        return self.green + self.red >= 150

    @property
    def valid(self):
        """Every cell free: a click here places the building."""
        return self.green >= 300 and self.red <= max(40, 0.03 * self.green)


def _center(mask, left, top):
    ys, xs = np.nonzero(mask)
    return (int(xs.mean()) + left, int(ys.mean()) + top) if len(xs) else None


def read_ghost(canvas, reference, point, window=WINDOW, region=None):
    """Ghost pixels near `point` (canvas pixels) that are absent from `reference`."""
    x, y = int(point[0]), int(point[1])
    rx, ry, rw, rh = view.tactical(region)
    left, right = max(rx, x - window[0]), min(rx + rw, x + window[0])
    top, bottom = max(ry, y - window[1]), min(ry + rh, y + window[1])
    if right <= left or bottom <= top:
        return Ghost(0, 0)
    now = canvas[top:bottom, left:right, :3].astype(np.int16)
    before = reference[top:bottom, left:right, :3].astype(np.int16)
    changed = np.abs(now - before).max(axis=2) > CHANGED
    green = changed & np.all(np.abs(now - GREEN) <= TOLERANCE, axis=2)
    red = changed & np.all(np.abs(now - RED) <= TOLERANCE, axis=2)
    return Ghost(int(green.sum()), int(red.sum()), _center(green, left, top), _center(red, left, top))
