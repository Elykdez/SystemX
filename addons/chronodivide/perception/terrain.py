"""Open ground near a point: where an MCV can deploy (a 4x4-cell construction yard).

The game shows no footprint before deploying, so spots are ranked by how much of a
yard-sized diamond around them looks like bare ground: grass and dirt are bright and
greener than blue. Shroud (black), water (bluish), structures (gray), and both house
colors are not ground. Trees and cliffs can pass as ground, so the deploy itself is
still verified and a blocked spot is never tried twice.
"""

import numpy as np

from .units import enemy_mask, house_mask
from .view import tactical

YARD = (120, 60)          # half width and height of a 4x4-cell diamond on screen


def ground_mask(canvas, region=None):
    region = tactical(region)
    x, y, w, h = region
    view = canvas[y:y + h, x:x + w].astype(np.int16)
    b, g, r = view[..., 0], view[..., 1], view[..., 2]
    ground = ((b + g + r) > 180) & (g >= b + 15)
    return ground & ~house_mask(canvas, region).astype(bool) & ~enemy_mask(canvas, region).astype(bool)


def openness(ground, point, half=YARD):
    """Fraction of ground pixels inside the diamond around `point` (0 when it leaves the view)."""
    x, y = point
    hw, hh = half
    height, width = ground.shape
    if x - hw < 0 or y - hh < 0 or x + hw > width or y + hh > height:
        return 0.0
    ys, xs = np.mgrid[-hh:hh + 1, -hw:hw + 1]
    inside = np.abs(xs) / hw + np.abs(ys) / hh <= 1.0
    patch = ground[y - hh:y + hh + 1, x - hw:x + hw + 1]
    return float(patch[inside].mean())


def open_spots(canvas, center, avoid=(), distances=(120, 180, 240), minimum=0.85, keep_out=60):
    """Spots around `center` whose yard-sized diamond is open ground, most open (then nearest) first."""
    ground = ground_mask(canvas)
    spots = []
    for distance in distances:
        for step in range(8):
            angle = step * np.pi / 4
            point = (int(center[0] + distance * np.cos(angle)), int(center[1] + distance * 0.5 * np.sin(angle)))
            if any(abs(point[0] - a[0]) + abs(point[1] - a[1]) < keep_out for a in avoid):
                continue
            score = openness(ground, point)
            if score >= minimum:
                spots.append((round(score, 2), -distance, point))
    return [point for _, _, point in sorted(spots, reverse=True)]
