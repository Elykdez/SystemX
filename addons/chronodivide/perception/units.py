"""Our house's objects in the tactical view: the MCV at start, then the base.

The camera start position varies per match (it stops at map edges), so the MCV is
found by its house-color stripes rather than assumed at the screen center. After
deployment, pressing H centers the base; the construction yard is then the main
cluster of our color, which anchors building placement. House colors come from the
client profile (colors.py; measured: our red, the opponent's blue), the tactical view
from the layout (view.py).
"""

import cv2
import numpy as np

from . import colors, view

TACTICAL = view.TACTICAL       # the 1024x768 default; readers use view.tactical() at call time
MCV_OFFSET = (6, 8)            # red stripe centroid -> clickable body center (measured)


def _crop(canvas, region):
    x, y, w, h = view.tactical(region)
    return canvas[y:y + h, x:x + w]


def house_mask(canvas, region=None):
    return colors.OWN.mask(_crop(canvas, region)).astype(np.uint8)


def enemy_mask(canvas, region=None, compressed=False):
    """The opponents' house colors (set in the skirmish setup; measured: blue)."""
    return colors.enemy_mask(_crop(canvas, region), compressed=compressed).astype(np.uint8)


def enemy_blobs(canvas, region=None, min_area=12, close=9, compressed=False):
    """Enemy units and buildings in view: [(area, (cx, cy), box)] largest first. A smaller `close`
    merges less: units driving side by side stay apart."""
    return [b for b in house_blobs(canvas, region, close=close, mask=enemy_mask(canvas, region, compressed=compressed))
            if b[0] >= min_area]


def house_blobs(canvas, region=None, close=9, mask=None):
    """[(area, (cx, cy), (x, y, w, h))] largest first, in canvas pixels."""
    region = view.tactical(region)
    mask = house_mask(canvas, region) if mask is None else mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    ox, oy = region[:2]
    blobs = [(int(stats[i, cv2.CC_STAT_AREA]), (int(centers[i][0]) + ox, int(centers[i][1]) + oy),
              (int(stats[i, 0]) + ox, int(stats[i, 1]) + oy, int(stats[i, 2]), int(stats[i, 3])))
             for i in range(1, count)]
    return sorted(blobs, reverse=True)


def find_mcv(canvas):
    """Click point on the undeployed MCV, or None. Expects it to be our only sizable red object."""
    blobs = house_blobs(canvas)
    if blobs and blobs[0][0] > 1000:
        return None                      # a deployed base is visible: not an MCV situation
    for area, center, (x, y, w, h) in blobs:
        if 120 <= area <= 900 and w <= 60 and h <= 45:
            return center[0] + MCV_OFFSET[0], center[1] + MCV_OFFSET[1]
    return None


def base_extent(canvas, near=None, radius=160):
    """Bounding box (x0, y0, x1, y1) of the main red cluster (construction yard and neighbors), or None."""
    blobs = [b for b in house_blobs(canvas) if b[0] >= 60]
    if not blobs:
        return None
    if near is not None:
        blobs = [b for b in blobs if abs(b[1][0] - near[0]) + abs(b[1][1] - near[1]) <= 2 * radius] or blobs
    seed = blobs[0]
    if seed[0] < 400:
        return None
    members = [b for b in blobs if abs(b[1][0] - seed[1][0]) <= radius and abs(b[1][1] - seed[1][1]) <= radius]
    return (min(b[2][0] for b in members), min(b[2][1] for b in members),
            max(b[2][0] + b[2][2] for b in members), max(b[2][1] + b[2][3] for b in members))


def find_base(canvas, near=None, radius=160):
    """Center of the main red cluster (construction yard and neighbors), or None."""
    extent = base_extent(canvas, near, radius)
    if extent is None:
        return None
    x0, y0, x1, y1 = extent
    return (x0 + x1) // 2, (y0 + y1) // 2
