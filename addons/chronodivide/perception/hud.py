"""Credits and power readouts from fixed HUD regions."""

from pathlib import Path

import numpy as np

GLYPH = (14, 10)   # rows, cols after padding a segmented digit


def yellowness(crop):
    c = crop.astype(np.int16)
    return np.clip(np.minimum(c[..., 1], c[..., 2]) - c[..., 0], 0, 255)


def segment(crop, threshold=230, compressed=False, color='yellow'):
    """Split the credits readout into digit glyphs by columns without solid ink.

    Anti-aliased edges bridge neighbouring digits, so segmentation uses only the
    solid glyph cores; classification then compares the full yellowness glyph.
    """
    ink = yellowness(crop)
    if color == 'blue':
        c = crop.astype(np.float32)
        ink = np.clip(c[..., 0] / 0.9, 0, 255)
    elif compressed:
        # JPEG subsamples color, smearing yellow across neighboring columns. Luminance keeps
        # the digit edges; a loose yellow gate excludes white HUD borders and gray background.
        c = crop.astype(np.float32)
        luminance = (0.114 * c[..., 0] + 0.587 * c[..., 1] + 0.299 * c[..., 2]) / 0.886
        ink = np.where(ink >= 40, np.clip(luminance, 0, 255), 0)
    mask = ink > threshold
    if color == 'blue':
        # JPEG sometimes joins Allied digits with one bright pixel. Split a wide core
        # at its shallow middle valley without trimming thin standalone digits.
        counts = mask.sum(axis=0)
        present = counts > 0
        starts = np.flatnonzero(present & ~np.r_[False, present[:-1]])
        ends = np.flatnonzero(present & ~np.r_[present[1:], False])
        for first, last in zip(starts, ends):
            if 8 <= last - first + 1 <= 14:
                middle = (first + last) / 2
                candidates = range(first + 3, last - 2)
                gap = min(candidates, key=lambda x: (counts[x], abs(x - middle)))
                if counts[gap] <= 1:
                    mask[:, gap] = False
    columns = np.flatnonzero(mask.any(axis=0))
    if not len(columns):
        return []
    rows = np.flatnonzero(mask.any(axis=1))
    top, bottom = rows[0], rows[-1] + 1
    glyphs, start = [], columns[0]
    for prev, col in zip(columns, list(columns[1:]) + [None]):
        if col is None or col != prev + 1:
            # Keep one anti-aliased column on each side; matching tolerates a 1-pixel shift.
            left, right = max(0, start - 1), min(ink.shape[1], prev + 2)
            glyph = ink[top:bottom, left:right].astype(np.float32) / 255
            padded = np.zeros(GLYPH, np.float32)
            h, w = min(GLYPH[0], glyph.shape[0]), min(GLYPH[1] - 1, glyph.shape[1])
            padded[:h, 1:1 + w] = glyph[:h, :w]
            glyphs.append((padded, prev + 1 - start))
            start = col
    return glyphs


def glyph_distance(a, b):
    return min(float(np.mean(np.abs(np.roll(a, shift, axis=1) - b))) for shift in (-1, 0, 1))


class CreditsReader:
    def __init__(self, templates=None, threshold=0.12, compressed=False, color='yellow'):
        self.templates = templates or {}          # digit -> list of (glyph, width)
        self.threshold = threshold
        self.compressed = compressed             # opt in for JPEG browser screenshots; native stays exact
        self.color = color
        self._cache = {}

    def _segment(self, crop):
        glyphs = segment(crop, threshold=215 if self.compressed else 230,
                         compressed=self.compressed, color=self.color)
        if self.compressed and self.templates:
            widest = max(w for entries in self.templates.values() for _, w in entries)
            if any(width > widest + 2 for _, width in glyphs):
                # JPEG can bridge adjacent cores (live 525 became a single wide "52").
                # Split with stricter ink; matching confidence and ambiguity checks stay unchanged.
                glyphs = segment(crop, threshold=230, compressed=True, color=self.color)
        return glyphs

    def learn(self, crop, value):
        glyphs = self._segment(crop)
        text = str(value)
        if len(glyphs) != len(text):
            return False
        for (glyph, width), digit in zip(glyphs, text):
            entries = self.templates.setdefault(digit, [])
            limit = 32 if self.color == 'blue' else 12
            if not any(np.array_equal(glyph, g) for g, _ in entries) and len(entries) < limit:
                entries.append((glyph, width))
        self._cache.clear()
        return True

    def read(self, crop):
        key = crop.tobytes()
        if key in self._cache:
            return self._cache[key]
        value = self._read(crop)
        if len(self._cache) > 2048:
            self._cache.clear()
        self._cache[key] = value
        return value

    def _read(self, crop):
        glyphs = self._segment(crop)
        if not glyphs or len(glyphs) > 7:
            return None
        digits = []
        width_tolerance = 2 if self.compressed else 1     # chroma blur can add a core column at either edge
        for glyph, width in glyphs:
            scores = sorted((min(glyph_distance(glyph, t) for t, w in entries
                                 if abs(w - width) <= width_tolerance), d)
                            for d, entries in self.templates.items()
                            if any(abs(w - width) <= width_tolerance for _, w in entries))
            best = scores[0] if scores else (1.0, None)
            if best[0] > self.threshold or best[1] is None:
                return None
            if self.compressed and len(scores) > 1 and scores[1][0] - best[0] < 0.01:
                return None                     # compression must not turn tied digits into guessed money
            digits.append(best[1])
        return int(''.join(digits))

    def save(self, path):
        arrays = {f'{d}_{i}_{w}': g for d, entries in self.templates.items() for i, (g, w) in enumerate(entries)}
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path, **kwargs):
        templates = {}
        with np.load(Path(path)) as data:
            for key in sorted(data.files):
                digit, _, width = key.split('_')
                templates.setdefault(digit, []).append((data[key], int(width)))
        return cls(templates, **kwargs)


def read_power(crop):
    """Power bar fill and level.

    Red fills from the bottom for consumption; yellow and green above it show headroom.
    A healthy reactor supplying a refinery can show yellow without any green. In the
    speed-6 recording of 2026-09-29, the game's explicit low-power warning accompanied
    a damaged reactor and an all-red bar, not the earlier yellow-over-red bar.

    Returns dict(fill=0..1 of lit segments, level='ok'|'low'|'none').
    """
    c = crop.astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    green = (g > 120) & (g > r + 40) & (g > b + 40)
    red = (r > 120) & (r > g + 50) & (r > b + 50)
    yellow = (r > 150) & (g > 150) & (b < 100)
    rows = crop.shape[0]
    lit_rows = (green | red | yellow).any(axis=1)
    fill = float(lit_rows.sum()) / rows
    if not lit_rows.any():
        return dict(fill=0.0, level='none')
    red_rows = red.any(axis=1)
    reds, greens = int(red_rows.sum()), int(green.any(axis=1).sum())
    yellows = yellow.any(axis=1)
    # Count yellow headroom above consumption, not a thin separator or gold below the red.
    red_top = int(np.flatnonzero(red_rows)[0]) if reds else rows
    spare_yellow = int(yellows[:red_top].sum())
    level = ('ok' if greens >= 3 or spare_yellow >= 6 else
             'low' if reds >= 4 else 'ok' if greens else 'none')
    return dict(fill=round(fill, 3), level=level, red_rows=reds, green_rows=greens,
                yellow_rows=int(yellows.sum()))
