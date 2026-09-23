"""Locate the game's own (software) arrow cursor, drawn into the canvas under pointer lock."""

from pathlib import Path

import cv2
import numpy as np

PROFILES = Path(__file__).resolve().parents[1] / 'profiles'
RADAR_BOX = (869, 47, 143, 113)
# Over the online radar the game draws a small pure-white arrow instead (live game 17: the parked
# cursor vanished for the finder when the radar came up). (dx, dy) from its tip, measured on a frame:
RADAR_ARROW = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3),
               (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (2, 5), (2, 6), (3, 7), (3, 8)]
# ... and pixels around it that are not white (so no white area passes for it)
RADAR_ARROW_OUTSIDE = [(-1, 0), (-1, 2), (-1, 4), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (1, 5), (3, 5),
                       (0, 6), (1, 6), (3, 6), (2, 7), (4, 8), (3, 9)]


def find_radar_arrow(canvas, region):
    """Tip of the small white radar cursor inside region (x, y, w, h), or None."""
    x0, y0, w, h = region
    height, width = canvas.shape[:2]
    # The shape reaches 1 px left, 5 px right and 9 px below its tip.
    ax, ay = max(0, x0 - 1), max(0, y0)
    area = canvas[ay:min(height, y0 + h + 10), ax:min(width, x0 + w + 6)]
    white = np.all(area >= 225, axis=2)
    pad = 12
    lit = np.pad(white, pad, constant_values=False)      # beyond the area: not part of the arrow
    dark = np.pad(~white, pad, constant_values=False)    # ... and not proof of a dark surround
    rows, cols = white.shape

    def shifted(grid, dx, dy):
        return grid[pad + dy:pad + dy + rows, pad + dx:pad + dx + cols]
    score = sum(shifted(lit, dx, dy).astype(np.int16) for dx, dy in RADAR_ARROW)
    outside = sum(shifted(dark, dx, dy).astype(np.int16) for dx, dy in RADAR_ARROW_OUTSIDE)
    good = (score >= len(RADAR_ARROW) - 1) & (outside >= len(RADAR_ARROW_OUTSIDE) - 2)
    ys, xs = np.nonzero(good)
    tips = [(int(x) + ax, int(y) + ay) for x, y in zip(xs, ys)
            if x0 <= x + ax < x0 + w and y0 <= y + ay < y0 + h]
    return tips[0] if len(tips) == 1 else None


class CursorFinder:
    """Masked template match; the hotspot is the arrow tip at the sprite's top-left.
    Over the online radar the small white arrow is looked for instead."""

    def __init__(self, sprite, mask, max_error=38.0, radar=RADAR_BOX):
        self.sprite = np.asarray(sprite, np.uint8)
        self.mask = np.asarray(mask, bool)
        self.max_error = max_error
        self.radar = radar
        self._mask3 = np.repeat(self.mask[..., None], 3, axis=2).astype(np.uint8)

    @classmethod
    def load(cls, path=PROFILES / 'cursor-arrow.npz', **kwargs):
        with np.load(Path(path)) as data:
            return cls(data['sprite'], data['mask'], **kwargs)

    def find(self, canvas, region=None):
        """(x, y) of the hotspot in canvas pixels, or None when no confident match."""
        found = self._find_arrow(canvas, region)
        if found is None and self.radar is not None:
            found = self._find_radar_arrow(canvas, region)
        return found

    def _find_arrow(self, canvas, region):
        x0, y0 = 0, 0
        image = canvas
        if region is not None:
            x0, y0, w, h = region
            image = canvas[y0:y0 + h, x0:x0 + w]
        sh, sw = self.sprite.shape[:2]
        if image.shape[0] < sh or image.shape[1] < sw:
            return None
        scores = cv2.matchTemplate(image, self.sprite, cv2.TM_SQDIFF, mask=self._mask3)
        _, _, location, _ = cv2.minMaxLoc(scores)
        best = scores[location[1], location[0]]
        error = float(np.sqrt(best / (self.mask.sum() * 3)))
        if not np.isfinite(error) or error > self.max_error:
            return None
        return x0 + location[0], y0 + location[1]

    def _find_radar_arrow(self, canvas, region):
        rx, ry, rw, rh = self.radar
        x0, y0, w, h = region if region is not None else (0, 0, canvas.shape[1], canvas.shape[0])
        left, top = max(rx, x0), max(ry, y0)
        right, bottom = min(rx + rw, x0 + w), min(ry + rh, y0 + h)
        if right - left < 8 or bottom - top < 12:
            return None
        return find_radar_arrow(canvas, (left, top, right - left, bottom - top))
