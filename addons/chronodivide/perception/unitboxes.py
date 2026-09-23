"""Units on screen as bounding boxes: our selected units (found by their health bars) and the
enemy's objects (found by their house color), with a recognized name where the game's own marks
allow it. CPU only, a few milliseconds per view.

A selected unit shows its health bar in a white frame, four rows tall: the frame, two rows of
pips, the frame. The frame's length is the unit's size class and does not change with damage
(measured 2026-09-27 on recorded games: 18 px for infantry, 36 px for vehicles); the lit pips
shrink as the unit is hurt (green, then yellow and red; empty pips are dark grey). The unit's
sprite stands right below its bar, and a transport (Flak Track) shows a strip of cargo pips
under its sprite: black-framed 2 px cells. So the frame says infantry or vehicle, the pips say
how hurt it is, the cargo strip says transport, and the sprite's box gives its size; which of
the unit types we built it is follows from those (see `recognize`).

Zero-shot detectors are no substitute here (probe 2026-09-27, Grounding DINO-B on a recorded
frame: 6 of about 16 infantry found, no tank, 77 ms on an RTX 4090); a game-specific detector
would have to be trained on labeled frames (`tools/units.py --export` writes them).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from . import colors
from .units import enemy_blobs
from .view import tactical

WHITE = 245                    # health bar frames are (252, 252, 252)
FRAME_KINDS = ((22, 'vehicle'), (12, 'infantry'))     # frame length -> size class (36 and 18 px unbroken)
HIDDEN_FRAME = 6               # a shorter visible piece of a frame, its end hidden by the next bar
VEHICLE_FRAME, LONGEST_FRAME = 36, 40
BUILDING_PIXELS = 100          # enemy color blobs this large are buildings (tanks show 30-80 px)
# Where the sprite stands below a bar: x from the bar's ends (vehicles) or center (infantry), y below it.
SPRITE = {'vehicle': ((-8, 12), (5, 44)), 'infantry': ((-11, 11), (4, 30))}


@dataclass(frozen=True)
class Unit:
    box: tuple                   # (x0, y0, x1, y1) canvas pixels, x1 and y1 exclusive
    side: str                    # 'ours' or 'enemy'
    kind: str                    # 'infantry', 'vehicle', 'unit' (size unknown) or 'building'
    health: float | None = None  # 0..1 from the health bar's pips
    bar: tuple | None = None     # (x, y, length) of the health bar frame
    cargo: bool = False          # a transport's cargo pips under the sprite
    name: str | None = None      # recognized item id (knowledge file), when known
    sure: bool = False           # the name is certain (the only candidate), else a best guess

    @property
    def center(self):
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2

    def label(self, items=None):
        """Short text for overlays: name (or kind), '?' for a guess, and health."""
        item = items.get(self.name) if items and self.name else None
        text = (item.get('short', item['name']) if item else self.name) or self.kind
        text += '' if self.sure or self.name is None else '?'
        return text if self.health is None else f'{text} {round(100 * self.health)}%'


def bar_frames(canvas, region=None, compressed=False):
    """Health bar frames of the selected units in view: [(x, y, length, health, kind)], one per
    unit; kind is 'infantry' or 'vehicle' by the frame's length."""
    x0, y0, w, h = tactical(region)
    view = canvas[y0:y0 + h, x0:x0 + w]
    # JPEG chroma subsampling bleeds green pips into the one-pixel white frame.
    white = (cv2.cvtColor(view, cv2.COLOR_BGR2GRAY) >= 225 if compressed
             else view.min(axis=2) >= WHITE)
    # A frame is white on its top row and again three rows below, with pips between.
    rows = (white[:-3] & white[3:]).astype(np.uint8)
    rows = cv2.morphologyEx(rows, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))   # lines crossing it
    # Horizontal runs, row by row (stacked bars touch each other's frames, so no 2D components).
    steps = np.diff(np.pad(rows, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    starts, ends = np.nonzero(steps == 1), np.nonzero(steps == -1)
    pieces = [(int(y), int(x), int(end - x)) for y, x, end in zip(starts[0], starts[1], ends[1]) if end - x >= 6]

    def edge(x, y):                       # a frame's end: white on both pip rows
        return 0 <= x < w and y + 2 < h and bool(white[y + 1, x] and white[y + 2, x])
    runs = []
    for y, x, length in pieces:
        last = runs[-1] if runs else None
        # One frame broken by an overlap (a badge, another bar): neither side of the gap is a frame's end.
        if last and last[1] == y and 0 <= x - (last[0] + last[2]) <= 12 and x + length - last[0] <= 40 and \
                not edge(last[0] + last[2] - 1, y) and not edge(x, y):
            last[2] = x + length - last[0]
            continue
        runs.append([x, y, length])
    found = []
    for x, y, length in runs:
        # A frame's ends are white on its pip rows; more ends inside a run are bars side by side, or
        # overlapping (the top one hides the other's end), so the run splits there. No end: not a bar.
        ends = [x + i for i in range(length) if edge(x + i, y)]
        if not ends or y + 3 >= h:
            continue
        cuts = sorted({x, *ends, x + length - 1})
        segments = [(a, b - a + 1) for a, b in zip(cuts, cuts[1:]) if b - a + 1 >= HIDDEN_FRAME]
        for index, (a, length) in enumerate(segments):
            inside = view[y + 1:y + 3, a + 1:a + length - 1].astype(np.int16)
            if inside.size == 0 or (inside.min(axis=2) >= WHITE).mean() > 0.3:
                continue                                         # white inside: not a bar
            health = pip_health(inside, compressed)
            if health is None:
                continue
            if length > LONGEST_FRAME:                               # vehicle bars touching end to end
                count = max(2, round(length / VEHICLE_FRAME))
                found += [(a + x0 + i * length // count, y + y0, length // count, health, 'vehicle')
                          for i in range(count)]
                continue
            if length >= FRAME_KINDS[-1][0]:
                kind = size_class(length)
            elif index + 1 < len(segments) and segments[index + 1][0] == a + length - 1:
                kind = size_class(max(segments[index + 1][1], FRAME_KINDS[-1][0]))   # hidden by its neighbor
            else:
                continue
            found.append((a + x0, y + y0, length, health, kind))
    return found


def pip_health(inside, compressed=False):
    """Share of lit pips inside a bar frame (None when the inside does not look like pips)."""
    b, g, r = inside[..., 0], inside[..., 1], inside[..., 2]
    lit = ((g >= 150) & (b <= 90)) | ((r >= 150) & (g <= 90) & (b <= 90))          # green, yellow, red pips
    dim = (g >= 60) & (r <= 40) & (b <= 40)                                       # the dark gap between them
    spread = inside.max(axis=2) - inside.min(axis=2)
    empty = (inside.max(axis=2) <= 70) & (spread <= 20)
    if compressed:
        lit = (((g >= 80) & (g - b >= 25) & (g - r >= 15)) |
               ((r >= 100) & (r - b >= 35) & ((r - g >= 25) | (g - b >= 35))))
        dim = np.zeros_like(lit)
        empty = (inside.max(axis=2) <= 100) & (spread <= 45)
    filled, total = int(lit.sum() + dim.sum()), int(lit.sum() + dim.sum() + empty.sum())
    if total < 0.6 * inside.shape[0] * inside.shape[1]:
        return None
    return round(filled / total, 2)


def size_class(length):
    return next(kind for limit, kind in FRAME_KINDS if length >= limit)


def cargo_strip(canvas, box):
    """A transport's cargo pips under a sprite: a black-framed strip of small cells."""
    x0, y0, x1, y1 = box
    view = canvas[max(0, y1 - 14):min(canvas.shape[0], y1 + 8), max(0, x0):min(canvas.shape[1], x1)]
    if view.shape[0] < 4 or view.shape[1] < 12:
        return False
    black = view.max(axis=2) <= 16
    lines = (black[:-3] & black[3:]).astype(np.uint8)            # frame rows three apart
    count, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=4)
    return any(stats[i, 3] == 1 and stats[i, 2] >= 12 for i in range(1, count))


def sprite_box(canvas, bar, kind):
    """The sprite's box below a health bar: the unit's own pixels (our house color, and the greys
    of armor and outline) in the window where it stands, tightened; the window when unclear."""
    x, y, length = bar
    (dx0, dx1), (dy0, dy1) = SPRITE[kind]
    left, right = (x + dx0, x + length + dx1) if kind == 'vehicle' else (x + length // 2 + dx0, x + length // 2 + dx1)
    height, width = canvas.shape[:2]
    window = (max(0, left), max(0, y + dy0), min(width, right), min(height, y + dy1))
    pixels = canvas[window[1]:window[3], window[0]:window[2]]
    if pixels.size == 0:
        return window
    own = colors.OWN.mask(pixels)
    wide = pixels.astype(np.int16)
    grey = (wide.max(axis=2) - wide.min(axis=2) <= 24) & (wide.max(axis=2) >= 36)
    mask = cv2.morphologyEx((own | grey).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return window
    # The part nearest the top center of the window (where the bar points), large enough to be a body.
    cx = (window[2] - window[0]) / 2
    parts = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 12]
    if not parts:
        return window
    main = min(parts, key=lambda i: abs(stats[i, 0] + stats[i, 2] / 2 - cx) + stats[i, 1] / 2)
    keep = [i for i in parts if i == main or _near(stats[i], stats[main], 3)]
    xs0 = min(stats[i, 0] for i in keep)
    ys0 = min(stats[i, 1] for i in keep)
    xs1 = max(stats[i, 0] + stats[i, 2] for i in keep)
    ys1 = max(stats[i, 1] + stats[i, 3] for i in keep)
    return (window[0] + int(xs0), window[1] + int(ys0), window[0] + int(xs1), window[1] + int(ys1))


def _near(a, b, gap):
    return not (a[0] > b[0] + b[2] + gap or b[0] > a[0] + a[2] + gap or
                a[1] > b[1] + b[3] + gap or b[1] > a[1] + a[3] + gap)


def our_units(canvas, region=None, compressed=False):
    """Our selected units in view, one per health bar: size class, health, sprite box, cargo."""
    units = []
    for x, y, length, health, kind in bar_frames(canvas, region, compressed):
        box = sprite_box(canvas, (x, y, length), kind)
        units.append(Unit(box, 'ours', kind, health, (x, y, length),
                          cargo=kind == 'vehicle' and cargo_strip(canvas, box)))
    return units


def enemy_objects(canvas, region=None, min_area=25):
    """The enemy's units and buildings in view, from its house color (largest first)."""
    return [Unit((x, y, x + w, y + h), 'enemy', 'building' if area >= BUILDING_PIXELS else 'unit')
            for area, _, (x, y, w, h) in enemy_blobs(canvas, region, min_area)]


def recognize(units, built, items, prefer=None):
    """Name our units from the types we built: a unit's size class (infantry or vehicle) and its
    cargo strip leave the types it can be; one left is certain, else the most numerous is a guess.

    `built` maps item ids to counts (combat units only); `items` is the knowledge file's items
    (a "cargo" item is a transport)."""
    named = []
    for unit in units:
        if unit.side != 'ours' or unit.kind not in ('infantry', 'vehicle'):
            named.append(unit)
            continue
        lane = 'infantry' if unit.kind == 'infantry' else 'vehicles'
        pool = {item: n for item, n in built.items() if n and items.get(item, {}).get('lane') == lane}
        if unit.kind == 'vehicle':
            pool = {i: n for i, n in pool.items() if bool(items[i].get('cargo')) == unit.cargo} or pool
        if not pool:
            named.append(unit)
            continue
        name = max(pool, key=lambda item: pool[item])
        named.append(Unit(unit.box, unit.side, unit.kind, unit.health, unit.bar, unit.cargo, name, len(pool) == 1))
    return named


def draw(image, units, items=None, scale=1.0, font=0.8, thickness=2):
    """Boxes and labels for developers: ours green (yellow when the name is a guess), the enemy's
    units red and buildings orange, each outlined in black so it stands out on any ground. Every
    box is drawn; a label only where it covers no other."""
    taken = []
    for unit in sorted(units, key=lambda u: (u.box[1], u.box[0])):
        x0, y0, x1, y1 = (int(round(v * scale)) for v in unit.box)
        color = ((0, 200, 0) if unit.sure or unit.name is None else (0, 210, 230)) if unit.side == 'ours' \
            else (0, 0, 230) if unit.kind != 'building' else (0, 90, 230)
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness + 2)
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
        text = unit.label(items) if unit.side == 'ours' else unit.kind
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font, 1)
        spot = (x0, max(height + 1, y0 - 2))
        area = (spot[0], spot[1] - height, spot[0] + width, spot[1] + 1)
        if any(area[0] < t[2] and t[0] < area[2] and area[1] < t[3] and t[1] < area[3] for t in taken):
            continue
        taken.append(area)
        cv2.putText(image, text, spot, cv2.FONT_HERSHEY_PLAIN, font, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, text, spot, cv2.FONT_HERSHEY_PLAIN, font, color, 1, cv2.LINE_AA)
    return image
