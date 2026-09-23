"""Labeled pixel templates with unknown rejection, generalized from the Minesweeper profile.

HUD sprites in Chrono Divide are drawn at a fixed UI scale, so a crop of a known
element matches its template almost exactly. A label is accepted only when the
best match is close enough AND clearly separated from every other label;
everything else is `None` (unknown), never a guess.
"""

from pathlib import Path

import cv2
import numpy as np


def prepare(image, size):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a BGR crop.')
    if image.shape[1::-1] != tuple(size):
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    return image.astype(np.int16)


def distance(a, b):
    """Mean absolute difference per channel, 0..1."""
    return float(np.mean(np.abs(a - b))) / 255


class TemplateSet:
    def __init__(self, name, size, threshold=0.06, margin=0.02):
        self.name, self.size = name, tuple(size)
        self.threshold, self.margin = threshold, margin
        self.samples = {}                  # label -> list[int16 arrays]
        self.revision = 0
        self._cache = {}

    def add(self, label, image, limit=24):
        sample = prepare(image, self.size)
        entries = self.samples.setdefault(label, [])
        if any(np.array_equal(sample, e) for e in entries):
            return False
        if len(entries) >= limit:
            entries.pop(0)
        entries.append(sample)
        self.revision += 1
        self._cache.clear()
        return True

    def scores(self, image):
        sample = prepare(image, self.size)
        return sorted((min(distance(sample, t) for t in templates), label)
                      for label, templates in self.samples.items() if templates)

    def classify(self, image):
        """(label | None, best distance, separation from the runner-up)."""
        key = image.tobytes() if image.shape[1::-1] == self.size else None
        if key is not None and key in self._cache:
            return self._cache[key]
        scores = self.scores(image)
        if not scores:
            return None, None, None
        best, label = scores[0]
        separation = scores[1][0] - best if len(scores) > 1 else 1.0
        result = (label if best <= self.threshold and separation >= self.margin else None, best, separation)
        if key is not None:
            if len(self._cache) >= 2048:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result
        return result

    def save(self, path):
        arrays = {f'{label}__{i}': sample.astype(np.uint8) for label, entries in self.samples.items()
                  for i, sample in enumerate(entries)}
        np.savez_compressed(path, __meta__=np.array([self.name, self.size[0], self.size[1],
                                                     self.threshold, self.margin], dtype=object), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=True) as data:
            name, width, height, threshold, margin = data['__meta__']
            result = cls(str(name), (int(width), int(height)), float(threshold), float(margin))
            for key in sorted(k for k in data.files if k != '__meta__'):
                label = key.rsplit('__', 1)[0]
                result.samples.setdefault(label, []).append(data[key].astype(np.int16))
        return result
