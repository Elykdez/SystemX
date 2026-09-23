"""Sidebar cameo reading: item identity plus state (available/disabled/building/ready).

Measured on Chrono Divide's Soviet sidebar at 1024x768:
- a disabled cameo is exactly 0.5 x the normal image;
- a cameo under construction is covered by a yellow veil (~53% blend with a light
  yellow) that shrinks clockwise, so the unveiled fraction is the build progress;
- "Ready" is yellow text over the top of the cameo; hover adds a yellow border.
Each template pixel may therefore appear normal, dark, or veiled. The item whose
template explains the most informative pixels wins, if clearly ahead of the rest.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DARK = 0.5
VEIL_ALPHA = 0.53
VEIL_COLOR = np.array([123.0, 218.0, 235.0])     # BGR, fitted from recorded frames
INSET = (3, 45, 3, 57)                           # rows/cols inside the hover border
READY_BAND = (4, 22, 12, 48)                     # where "Ready" text is drawn
COUNT_BAND = (3, 16, 49, 59)                     # queued-units count, yellow digits at the top right


@dataclass
class SlotReading:
    item: str | None
    state: str              # empty | available | disabled | building | ready | unknown | menu
    progress: float | None = None
    score: float = 0.0
    margin: float = 0.0
    queued: bool = False    # a queue count is drawn: units of this kind are waiting or in production


def inner(image):
    top, bottom, left, right = INSET
    return image[top:bottom, left:right].astype(np.float32)


def yellow_mask(crop, band=READY_BAND, compressed=False, color='yellow'):
    top, bottom, left, right = band
    band = crop[top:bottom, left:right].astype(np.int16)
    b, g, r = band[..., 0], band[..., 1], band[..., 2]
    if color == 'blue':
        return (b > 190) & (g > 150) & (b - r > 30)
    if compressed:
        return (r > 200) & (g > 190) & (np.minimum(r, g) - b > 60)
    return (r > 200) & (g > 190) & (b < 110)


def count_mark(crop, min_pixels=10, compressed=False, color='yellow'):
    """The yellow queue count (1, 2, 3...) drawn at a unit cameo's top right; measured ~21 pixels."""
    if color == 'blue':
        mask = yellow_mask(crop, COUNT_BAND, compressed=compressed, color=color)
    elif compressed:
        top, bottom, left, right = COUNT_BAND
        band = crop[top:bottom, left:right].astype(np.int16)
        b, g, r = band[..., 0], band[..., 1], band[..., 2]
        mask = (r > 165) & (g > 155) & (np.minimum(r, g) - b > 60)
    else:
        mask = yellow_mask(crop, COUNT_BAND)
    return int(np.count_nonzero(mask)) >= min_pixels


def yellow_text(crop):
    return int(np.count_nonzero(yellow_mask(crop)))


SHIFTS = tuple((dx, dy) for dx in range(-3, 4) for dy in range(-2, 3))


def shifted_iou(mask, template, shifts=SHIFTS):
    best = 0.0
    for dx, dy in shifts:
        moved = np.roll(np.roll(template, dy, axis=0), dx, axis=1)
        union = np.count_nonzero(mask | moved)
        if union:
            best = max(best, np.count_nonzero(mask & moved) / union)
    return best


class CameoCatalog:
    def __init__(self, templates, empty=None, tolerance=26.0, veil_tolerance=34.0,
                 min_score=0.72, min_margin=0.12, ready_pixels=40, ready_masks=(), ready_iou=0.6,
                 compressed=False, veil_color=None, text_color='yellow'):
        self.templates = {k: np.asarray(v, np.uint8) for k, v in templates.items()}
        self.empty = None if empty is None else np.asarray(empty, np.uint8)
        self.ready_masks = [np.asarray(m, bool) for m in ready_masks]
        self.ready_iou = ready_iou
        self.compressed = compressed
        self.veil_color = VEIL_COLOR if veil_color is None else np.asarray(veil_color, np.float32)
        self.text_color = text_color
        self.tolerance, self.veil_tolerance = tolerance, veil_tolerance
        self.min_score, self.min_margin, self.ready_pixels = min_score, min_margin, ready_pixels
        self._prepared = {}
        for item, template in self.templates.items():
            self._prepare(item, template)
        self._cache = {}

    def _prepare(self, item, template):
        t = inner(template)
        informative = t.max(axis=2) >= 56       # dark pixels look alike in every state
        veil = (1 - VEIL_ALPHA) * t + VEIL_ALPHA * self.veil_color
        self._prepared[item] = (t, DARK * t, veil, informative)

    def teach(self, item, crop):
        """Learn an item from a cameo shown in its normal look (available: not dark, veiled, or Ready);
        item None learns the empty slot."""
        crop = np.asarray(crop, np.uint8).copy()
        if item is None:
            self.empty = crop
        else:
            self.templates[item] = crop
            self._prepare(item, crop)
        self._cache.clear()

    @classmethod
    def load(cls, path, **kwargs):
        with np.load(Path(path)) as data:
            templates = {k[5:]: data[k] for k in data.files if k.startswith('item:')}
            empty = data['empty'] if 'empty' in data.files else None
            ready = [data[k] for k in sorted(data.files) if k.startswith('ready:')]
        return cls(templates, empty, ready_masks=ready, **kwargs)

    def save(self, path):
        arrays = {f'item:{k}': v for k, v in self.templates.items()}
        if self.empty is not None:
            arrays['empty'] = self.empty
        arrays.update({f'ready:{i}': m for i, m in enumerate(self.ready_masks)})
        np.savez_compressed(path, **arrays)

    def is_ready(self, crop):
        """The word "Ready" drawn over the cameo (not tooltip text or yellow artwork)."""
        mask = yellow_mask(crop, compressed=self.compressed, color=self.text_color)
        if np.count_nonzero(mask) < self.ready_pixels:
            return False
        if not self.ready_masks:
            return True
        return max(shifted_iou(mask, m) for m in self.ready_masks) >= self.ready_iou

    def explain(self, crop, item, dx=0):
        """Fractions of informative pixels explained as normal / dark / veiled for one template.

        `dx` compares the crop shifted horizontally: the two sidebar columns draw their
        icons one pixel apart, so a template learned in one column is shifted in the other.
        """
        c = inner(np.roll(crop, -dx, axis=1) if dx else crop)
        t, dark, veil, informative = self._prepared[item]
        normal = np.abs(c - t).max(axis=2) <= self.tolerance
        darkened = ~normal & (np.abs(c - dark).max(axis=2) <= self.tolerance)
        veiled = ~normal & ~darkened & (np.abs(c - veil).max(axis=2) <= self.veil_tolerance)
        total = max(1, int(informative.sum()))
        return (float((normal & informative).sum()) / total, float((darkened & informative).sum()) / total,
                float((veiled & informative).sum()) / total)

    def is_empty(self, crop):
        if self.empty is None:
            return False
        difference = np.abs(inner(crop) - inner(self.empty)).max(axis=2)
        return float(np.mean(difference <= self.tolerance)) >= 0.8

    def queued(self, crop, item, dx=0):
        if self.text_color != 'blue':
            return count_mark(crop, compressed=self.compressed)
        top, bottom, left, right = COUNT_BAND
        c = (np.roll(crop, -dx, axis=1) if dx else crop)[top:bottom, left:right].astype(np.float32)
        t = self.templates[item][top:bottom, left:right].astype(np.float32)
        b, g, r = c[..., 0], c[..., 1], c[..., 2]
        ink = (b > 215) & (g > 195) & (r > 140) & (b - r > 30) & (b - r < 85)
        # Allied sky artwork shares the numeral's hue; require ink unexplained by the cameo.
        difference = np.minimum.reduce([np.abs(c - variant).max(axis=2) for variant in
                                       (t, DARK * t, (1 - VEIL_ALPHA) * t + VEIL_ALPHA * self.veil_color)])
        return int(np.count_nonzero(ink & (difference > self.tolerance))) >= 6

    def classify(self, crop):
        key = crop.tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._classify(crop)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = result
        return result

    def _classify(self, crop):
        if self.is_empty(crop):
            return SlotReading(None, 'empty', score=1.0)
        shifts = {item: 0 for item in self.templates}
        scores = sorted(((sum(self.explain(crop, item)), item) for item in self.templates), reverse=True)
        if not scores:
            return SlotReading(None, 'unknown')
        if scores[0][0] < self.min_score:
            # Retry the leading candidates one pixel left and right (column offset).
            rescored = {}
            for score, candidate in scores[:3]:
                best_shift = max((sum(self.explain(crop, candidate, dx)), dx) for dx in (-1, 0, 1))
                rescored[candidate] = best_shift[0]
                shifts[candidate] = best_shift[1]
            scores = sorted(((rescored.get(item, score), item) for score, item in scores), reverse=True)
        best, item = scores[0]
        margin = best - (scores[1][0] if len(scores) > 1 else 0.0)
        if best < self.min_score or margin < self.min_margin:
            # Ready text or a tooltip can hide part of a cameo; accept a clear winner with Ready text.
            if not (best >= 0.5 and margin >= self.min_margin and self.is_ready(crop)):
                return SlotReading(None, 'unknown', score=best, margin=margin)
        normal, dark, veiled = self.explain(crop, item, shifts[item])
        if self.is_ready(crop):
            state, progress = 'ready', 1.0
        # Allied's final wedge is about 4%; idle Soviet artwork can explain that much veil.
        elif veiled >= (0.035 if self.text_color == 'blue' else 0.06):
            state, progress = 'building', round(normal / max(1e-6, normal + veiled), 3)
        elif dark > normal:
            state, progress = 'disabled', None
        else:
            state, progress = 'available', None
        return SlotReading(item, state, progress, round(best, 3), round(margin, 3),
                           self.queued(crop, item, shifts[item]))


def read_slots(canvas, layout, catalog):
    return [catalog.classify(box.crop(canvas)) for box in layout.grid.boxes()]


def lane_of(readings, knowledge):
    """The visible tab, inferred from which lane its recognized items belong to."""
    lanes = {knowledge['items'][r.item]['lane'] for r in readings if r.item in knowledge['items']}
    return lanes.pop() if len(lanes) == 1 else None


def annotate(image, layout, readings):
    """Draw slot readings on a BGR copy for the controller preview."""
    colors = {'available': (80, 220, 80), 'disabled': (120, 120, 120), 'building': (0, 200, 255),
              'ready': (255, 255, 0), 'unknown': (0, 0, 255), 'empty': (60, 60, 60)}
    for box, reading in zip(layout.grid.boxes(), readings):
        color = colors.get(reading.state, (255, 0, 255))
        cv2.rectangle(image, (box.x, box.y), (box.x + box.w - 1, box.y + box.h - 1), color, 1)
        if reading.state not in ('empty',):
            label = reading.state[:5] if reading.item is None else reading.item[:9]
            if reading.state == 'building' and reading.progress is not None:
                label += f' {reading.progress:.0%}'
            cv2.putText(image, label, (box.x + 2, box.y + 11), cv2.FONT_HERSHEY_PLAIN, .8, color, 1)
    return image
