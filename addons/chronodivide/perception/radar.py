"""Radar (minimap): map outline, camera box, our base and army, enemy sightings.

Measured on Chrono Divide's 1024x768 sidebar (a recorded game, 2026-09-23): once a Radar
Tower has power, the radar box shows the minimap. Shroud is black; the map outline and
the camera's view are pure yellow (BGR 0, 255, 255) rectangles; our objects are red dots
(BGR 24, 24, 255), the opponent's blue dots (BGR 212, 105, 34); explored ground keeps its
terrain colors. Offline, the box shows the faction emblem (or white static while the radar
comes online): its gold star is yellow too, but it has no long straight yellow lines.

The minimap is the tactical view scaled (x and y by different factors), so the camera box
is a rectangle and a radar point maps linearly to the map. Everything here is in canvas
pixels, so radar points can be clicked directly.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import colors

MIN_LINE = 30                # outline edges are at least this long (radar pixels)


def yellow_mask(crop, compressed=False):
    b, g, r = (crop[..., i].astype(np.int16) for i in range(3))
    if compressed:
        return (g > 200) & (r > 200) & (np.minimum(g, r) - b > 60)
    return (b < 80) & (g > 200) & (r > 200)


def own_mask(crop, compressed=False):
    """Our dots (measured: red), in the house color of the client profile."""
    return _house_mask(crop, colors.OWN, compressed)


def enemy_mask(crop, compressed=False):
    """The opponents' dots (measured: blue, G about 0.49 B and R about 0.16 B, the same ramp as in
    the tactical view)."""
    mask = np.zeros(crop.shape[:2], bool)
    for color in colors.ENEMIES:
        mask |= _house_mask(crop, color, compressed)
    return mask & ~ally_mask(crop, compressed)


def ally_mask(crop, compressed=False):
    mask = np.zeros(crop.shape[:2], bool)
    for color in colors.ALLIES:
        mask |= _house_mask(crop, color, compressed)
    return mask


def _house_mask(crop, color, compressed):
    if not compressed or color.hue is not None:
        return color.radar_mask(crop)
    # JPEG blends tiny dots with terrain: their dominant hue survives the exact RGB ramp.
    b, g, r = (crop[..., i].astype(np.int16) for i in range(3))
    if color.name == 'red':
        # Red units overlapping blue enemies become purple after JPEG chroma blending.
        # Keep strong red over green and positive red over blue to reject blue-only dots.
        mask = (r > 100) & (r - g > 55) & (r - b > 15)
        # Magenta waypoint lines bleed into this hue too; compact mixed unit dots remain.
        magenta = ((b > 60) & (r > 80) & (np.minimum(b, r) - g > 15) &
                   (np.abs(r - b) < 55)).astype(np.uint8)
        lines = np.zeros_like(magenta)
        segments = cv2.HoughLinesP(magenta, 1, np.pi / 180, 3, minLineLength=3, maxLineGap=1)
        # Compact overlapping red/blue dots can share the hue. Filter only with a long route outline present.
        route_visible = (segments is not None and any(
            np.hypot(x1 - x0, y1 - y0) >= 20 for x0, y0, x1, y1 in segments[:, 0]))
        if route_visible:
            for x0, y0, x1, y1 in segments[:, 0]:
                cv2.line(lines, (int(x0), int(y0)), (int(x1), int(y1)), 1, 3)
        return mask & ~cv2.dilate(lines, np.ones((5, 5), np.uint8)).astype(bool)
    if color.name == 'blue':
        return (b > 90) & (b - r > 40) & (b - g > 18)
    return color.radar_mask(crop)


def explored_mask(crop):
    return crop.astype(np.int16).sum(axis=2) > 40


def longest_runs(mask):
    """Longest run of consecutive True values in each row."""
    best = np.zeros(mask.shape[0], np.int32)
    run = np.zeros(mask.shape[0], np.int32)
    for column in mask.T.astype(np.int32):
        run = (run + 1) * column
        np.maximum(best, run, out=best)
    return best


def clusters(mask, offset, max_area=600, close=3):
    """[(area, (x, y))] largest first; blobs larger than max_area are terrain, not objects."""
    if not mask.any():
        return []
    joined = cv2.dilate(mask.astype(np.uint8), np.ones((close, close), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    found = []
    for i in range(1, count):
        pixels = (labels == i) & mask
        area = int(pixels.sum())
        if not area or area > max_area:
            continue
        ys, xs = np.nonzero(pixels)
        found.append((area, (int(round(xs.mean())) + offset[0], int(round(ys.mean())) + offset[1])))
    return sorted(found, reverse=True)


def distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


@dataclass(frozen=True)
class Radar:
    online: bool
    bounds: tuple | None = None          # map outline (x0, y0, x1, y1), inclusive
    view: tuple | None = None            # camera box (x0, y0, x1, y1)
    own: tuple = ()                      # ((area, (x, y)), ...) largest first
    enemy: tuple = ()
    explored_share: float = 0.0          # of the map area
    explored: np.ndarray | None = field(default=None, repr=False, compare=False)
    origin: tuple = (0, 0)               # canvas position of the radar box
    allies: tuple = ()                   # friendly objects, excluded from army strength and targets

    @property
    def center(self):
        x0, y0, x1, y1 = self.bounds
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def view_center(self):
        if self.view is None:
            return None
        x0, y0, x1, y1 = self.view
        return (x0 + x1) // 2, (y0 + y1) // 2

    def inside(self, point, margin=3):
        x0, y0, x1, y1 = self.bounds
        return (min(max(round(point[0]), x0 + margin), x1 - margin),
                min(max(round(point[1]), y0 + margin), y1 - margin))

    def in_view(self, point, margin=1):
        if self.view is None:
            return False
        x0, y0, x1, y1 = self.view
        return x0 - margin <= point[0] <= x1 + margin and y0 - margin <= point[1] <= y1 + margin

    def to_tactical(self, point, tactical):
        """Where a radar point is in the tactical view, when the camera box contains it."""
        x0, y0, x1, y1 = self.view
        fx = (point[0] - x0 + 0.5) / max(1, x1 - x0 + 1)
        fy = (point[1] - y0 + 0.5) / max(1, y1 - y0 + 1)
        return round(tactical.x + fx * tactical.w), round(tactical.y + fy * tactical.h)

    def explored_near(self, point, radius=6):
        """Share of explored (non-shroud) radar pixels around a point."""
        if self.explored is None:
            return 0.0
        x, y = round(point[0]) - self.origin[0], round(point[1]) - self.origin[1]
        height, width = self.explored.shape
        patch = self.explored[max(0, y - radius):min(height, y + radius + 1), max(0, x - radius):min(width, x + radius + 1)]
        return float(patch.mean()) if patch.size else 0.0

    def enemy_near(self, point, radius):
        return [e for e in self.enemy if distance(e[1], point) <= radius]

    def army_center(self, home, home_radius=14):
        """Area-weighted center of our objects away from home (the army in the field), or None."""
        away = [(a, p) for a, p in self.own if home is None or distance(p, home) > home_radius]
        area = sum(a for a, _ in away)
        if area < 3:
            return None
        return (round(sum(a * p[0] for a, p in away) / area), round(sum(a * p[1] for a, p in away) / area))

    def guesses(self, home):
        """Likely enemy start positions for a two-player map: our base mirrored through the map
        center (rotational symmetry first, then left-right and top-bottom mirrors)."""
        if home is None or self.bounds is None:
            return []
        cx, cy = self.center
        points = [(2 * cx - home[0], 2 * cy - home[1]), (2 * cx - home[0], home[1]), (home[0], 2 * cy - home[1])]
        return [self.inside(p) for p in points if distance(p, home) >= 20 and
                not any(a >= 12 and distance(p, friendly) < 14 for a, friendly in self.allies)]

    def unexplored_spot(self, home, cell=10):
        """Center of the most shrouded map area, preferring spots far from home, or None."""
        if self.explored is None or self.bounds is None:
            return None
        x0, y0, x1, y1 = self.bounds
        best = None
        for y in range(y0 + cell // 2, y1 - cell // 2 + 1, cell):
            for x in range(x0 + cell // 2, x1 - cell // 2 + 1, cell):
                hidden = 1.0 - self.explored_near((x, y), cell // 2)
                if any(a >= 12 and distance((x, y), friendly) < 14 for a, friendly in self.allies):
                    continue
                if hidden < 0.6:
                    continue
                score = hidden + (distance((x, y), home) / 200 if home else 0)
                if best is None or score > best[0]:
                    best = (score, (x, y))
        return best[1] if best else None

    def summary(self):
        return dict(online=self.online, bounds=self.bounds, view=self.view,
                    own=[list(p) + [a] for a, p in self.own[:6]], enemy=[list(p) + [a] for a, p in self.enemy[:8]],
                    allies=[list(p) + [a] for a, p in self.allies[:8]],
                    explored=round(self.explored_share, 2))


def read_radar(canvas, box, compressed=False, line_color='yellow'):
    """Radar reading from a canvas (BGR) and the radar region (layout Box)."""
    crop = box.crop(canvas)
    if line_color == 'white':
        channels = crop.astype(np.int16)
        yellow = (channels.min(axis=2) > 150) & (np.ptp(channels, axis=2) < 60)
        # Native Allied captures use pale blue for both outlines; browser fixtures use white.
        yellow |= np.abs(channels - np.array([254, 211, 166])).max(axis=2) <= 20
        # The Allied sidebar frame also has pale edges beside the minimap.
        yellow[:, -4:] = False
        yellow[-2:, :] = False
    else:
        yellow = yellow_mask(crop, compressed)
    rows = np.flatnonzero(longest_runs(yellow) >= MIN_LINE)
    cols = np.flatnonzero(longest_runs(yellow.T) >= MIN_LINE)
    if len(rows) < 2 or len(cols) < 2:
        return Radar(False, origin=(box.x, box.y))
    if line_color == 'white':
        rows = np.array([y for y in rows if yellow[y, cols[0]:cols[-1] + 1].mean() >= 0.8])
        if len(rows) < 2:
            return Radar(False, origin=(box.x, box.y))
    top, bottom, left, right = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    if bottom - top < 20 or right - left < 20:
        return Radar(False, origin=(box.x, box.y))
    if line_color == 'white' and max(right - left, bottom - top) > 2 * min(right - left, bottom - top):
        # The Allied activation flash makes a wide emblem band, not the minimap frame.
        return Radar(False, origin=(box.x, box.y))
    # A true outline is closed: its edges span (almost) the whole rectangle.
    edges = [yellow[top, left:right + 1], yellow[bottom, left:right + 1],
             yellow[top:bottom + 1, left], yellow[top:bottom + 1, right]]
    if min(float(edge.mean()) for edge in edges) < 0.8:
        return Radar(False, origin=(box.x, box.y))
    ox, oy = box.x, box.y
    inner = yellow.copy()
    inner[[top, bottom], :] = False
    inner[:, [left, right]] = False
    inner[:top] = inner[bottom + 1:] = False
    inner[:, :left] = inner[:, right + 1:] = False
    if line_color == 'white':
        # Reject pale terrain pixels; camera edges form straight runs.
        horizontal = cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 5), np.uint8))
        vertical = cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
        inner &= (horizontal | vertical).astype(bool)
    view = None
    ys, xs = np.nonzero(inner)
    if len(xs) >= 8:
        vx0, vx1, vy0, vy1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        # A camera box at the map edge shares that edge with the outline.
        vx0 = left if vx0 <= left + 2 else vx0
        vy0 = top if vy0 <= top + 2 else vy0
        vx1 = right if vx1 >= right - 2 else vx1
        vy1 = bottom if vy1 >= bottom - 2 else vy1
        if vx1 - vx0 >= 4 and vy1 - vy0 >= 4:
            view = (vx0 + ox, vy0 + oy, vx1 + ox, vy1 + oy)
    area = np.zeros(yellow.shape, bool)
    area[top + 1:bottom, left + 1:right] = True
    area &= ~yellow
    explored = explored_mask(crop) & area
    return Radar(True, (left + ox, top + oy, right + ox, bottom + oy), view,
                 tuple(clusters(own_mask(crop, compressed) & area, (ox, oy))),
                 tuple(clusters(enemy_mask(crop, compressed) & area, (ox, oy))),
                 float(explored.sum()) / max(1, int(area.sum())), explored, (ox, oy),
                 allies=tuple(clusters(ally_mask(crop, compressed) & area, (ox, oy))))


def annotate(image, radar, target=None, home=None):
    """Draw a radar reading on a BGR copy of the canvas (controller preview)."""
    if radar is None or not radar.online:
        return image
    x0, y0, x1, y1 = radar.bounds
    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 0), 1)
    if radar.view:
        cv2.rectangle(image, radar.view[:2], radar.view[2:], (255, 0, 255), 1)
    for _, point in radar.enemy:
        cv2.circle(image, point, 3, (255, 128, 0), 1)
    if home is not None:
        cv2.drawMarker(image, tuple(map(int, home)), (0, 255, 0), cv2.MARKER_DIAMOND, 8, 1)
    if target is not None:
        cv2.drawMarker(image, tuple(map(int, target)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
    return image
