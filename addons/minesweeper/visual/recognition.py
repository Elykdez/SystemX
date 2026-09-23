from dataclasses import dataclass
from pathlib import Path
import json
import math

import cv2
import numpy as np

from ..board import Board, Box, Cell


LABELS = ('hidden', 'flag', 'mine', *map(str, range(9)))


@dataclass(frozen=True)
class Grid:
    x: int
    y: int
    width: int
    height: int
    rows: int
    cols: int
    mines: int | None = None

    def __post_init__(self):
        if any(type(v) is not int for v in (self.x, self.y, self.width, self.height, self.rows, self.cols)):
            raise ValueError('Grid values must be integers.')
        if min(self.x, self.y) < 0 or not (1 <= self.rows <= 32 and 1 <= self.cols <= 64):
            raise ValueError('Invalid grid origin or dimensions.')
        if min(self.width / self.cols, self.height / self.rows) < 12 or self.rows * self.cols > 1024:
            raise ValueError('Cells must be at least 12 pixels; at most 1024 cells are supported.')
        if self.mines is not None and (type(self.mines) is not int or not 0 <= self.mines < self.rows * self.cols):
            raise ValueError('Invalid total mines.')

    def boxes(self):
        for row in range(self.rows):
            for col in range(self.cols):
                x0, x1 = [round(self.x + c * self.width / self.cols) for c in (col, col + 1)]
                y0, y1 = [round(self.y + r * self.height / self.rows) for r in (row, row + 1)]
                yield row, col, Box(x0, y0, x1 - x0, y1 - y0)

    def crop(self, image, box, origin=(0, 0)):
        x, y, w, h = map(int, (box.x, box.y, box.width, box.height))
        x, y = x - origin[0], y - origin[1]
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            raise ValueError('Board is clipped. Recalibrate on the complete board.')
        return image[y:y + h, x:x + w]


def normalize(image):
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
        raise ValueError('Cell sample must be a BGR image at least 8 pixels wide and tall.')
    return cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.uint8)


def features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    interior = image[5:27, 5:27].astype(np.float32)
    mask = np.max(np.abs(interior - np.median(interior.reshape(-1, 3), axis=0)), axis=2) > 45
    return cv2.Sobel(gray, cv2.CV_32F, 1, 0), interior, mask


def distance(a, b, fa=None, fb=None):
    # Keep bevels as well as glyphs: covered and empty cells can share a background.
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255
    color = float(np.mean(delta))
    fa, fb = fa if fa is not None else features(a), fb if fb is not None else features(b)
    edge = float(np.mean(np.abs(fa[0] - fb[0]))) / 1020
    union = fa[2] | fb[2]
    if np.any(union):
        shape_error = float(np.mean(fa[2][union] != fb[2][union]))
        ink_error = float(np.mean(np.abs(fa[1][union] - fb[1][union]))) / 255
    else:
        shape_error = ink_error = 0
    return color + .3 * edge + .15 * shape_error + .15 * ink_error


class Profile:
    def __init__(self, name='Custom', threshold=.075, margin=.012):
        self.name, self.threshold, self.margin = name, threshold, margin
        self.samples = {}
        self.revision = 0
        self._features = {}
        self._classifications = {}

    def add(self, label, image):
        if label not in LABELS:
            raise ValueError('Unsupported cell label.')
        entries = self.samples.setdefault(label, [])
        sample = normalize(image)
        if any(np.array_equal(sample, existing) for existing in entries):
            return
        if len(entries) >= 32:
            raise ValueError('At most 32 samples per label; use a separate profile for another theme.')
        entries.append(sample)
        self._features[id(entries[-1])] = features(entries[-1])
        self.revision += 1
        self._classifications.clear()

    def classify(self, image):
        sample = normalize(image)
        # Cache identical pixels, never board positions: every frame is still inspected.
        key = (self.threshold, self.margin, sample.tobytes())
        if key in self._classifications:
            return self._classifications[key]
        prepared = features(sample)
        scores = sorted((min(distance(sample, t, prepared, self._features[id(t)]) for t in templates), label)
                        for label, templates in self.samples.items() if templates)
        if not scores:
            return None, None, None
        best, label = scores[0]
        separation = scores[1][0] - best if len(scores) > 1 else 1.0
        result = (None if best > self.threshold or separation < self.margin else label, best, separation)
        if len(self._classifications) >= 512:
            self._classifications.pop(next(iter(self._classifications)))
        self._classifications[key] = result
        return result

    def save(self, path):
        Path(path).write_text(json.dumps(dict(version=1, name=self.name, threshold=self.threshold,
            margin=self.margin, samples={k: [i.tolist() for i in v] for k, v in self.samples.items()})), encoding='utf-8')

    @classmethod
    def load(cls, path):
        if Path(path).stat().st_size > 8_000_000:
            raise ValueError('Profile is too large.')
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        if raw.get('version') != 1:
            raise ValueError('Unsupported profile version.')
        for key in ('threshold', 'margin'):
            value = raw[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < .5:
                raise ValueError('Invalid recognition threshold.')
        result = cls(str(raw['name']), raw['threshold'], raw['margin'])
        for label, entries in raw['samples'].items():
            for entry in entries:
                array = np.asarray(entry)
                if array.shape != (32, 32, 3) or array.dtype.kind not in 'iu' or array.min() < 0 or array.max() > 255:
                    raise ValueError('Invalid profile sample.')
                result.add(label, array.astype(np.uint8))
        return result


def recognize(image, grid, profile, origin=(0, 0)):
    cells, details = [], []
    for row, col, box in grid.boxes():
        label, error, margin = profile.classify(grid.crop(image, box, origin))
        details.append(dict(row=row, col=col, label=label, error=error, margin=margin))
        if label is not None:
            cells.append(Cell(row, col, label, box))
    if len(cells) != grid.rows * grid.cols:
        return None, details
    status = 'playing'
    if any(c.state == 'mine' for c in cells):
        status = 'lost'
    elif grid.mines is not None and sum(c.state.isdigit() for c in cells) == grid.rows * grid.cols - grid.mines:
        status = 'won'
    return Board(grid.rows, grid.cols, tuple(cells), status, grid.mines), details


def suggest_grid(image, rows, cols, mines=None):
    """Suggest a regular grid from repeated cell rectangles; user confirms it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    groups = {}
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 12 <= w <= 100 and 12 <= h <= 100 and .7 < w / h < 1.4:
            groups.setdefault((w, h), set()).add((x, y))
    for (w, h), points in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        if len(points) < max(4, rows * cols * .4):
            continue
        xs, ys = sorted({x for x, _ in points}), sorted({y for _, y in points})
        if len(xs) != cols or len(ys) != rows or len(xs) < 2 or len(ys) < 2:
            continue
        dx, dy = np.diff(xs), np.diff(ys)
        if np.ptp(dx) <= 1 and np.ptp(dy) <= 1 and max(dx) <= w + 5 and max(dy) <= h + 5:
            pitch_x, pitch_y = round(float(np.median(dx))), round(float(np.median(dy)))
            # Canny commonly traces the inner bevel; include the full cell pitch.
            return Grid(xs[0] + w - pitch_x, ys[0] + h - pitch_y, pitch_x * cols,
                        pitch_y * rows, rows, cols, mines)
    raise ValueError('No unambiguous grid found. Drag around the board in the preview.')
