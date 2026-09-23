"""Visible resource crystals at native scale; abstain on depleted ground."""

import cv2
import numpy as np

from . import colors
from .view import tactical


def _ore_chips(canvas, region=None):
    if canvas is None:
        return []
    x, y, w, h = tactical(region)
    crop = canvas[y:y + h, x:x + w]
    b, g, r = (crop[..., i].astype(np.int16) for i in range(3))
    gold = ((r >= 235) & (g >= 210) & (b <= 160) & (r - b >= 90)).astype(np.uint8)
    occupied = colors.OWN.mask(crop) | colors.enemy_mask(crop, compressed=True)
    occupied = cv2.dilate(occupied.astype(np.uint8), np.ones((31, 31), np.uint8))
    gold[occupied != 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gold, connectivity=8)
    chips = []
    for i in range(1, count):
        left, top, width, height, area = stats[i]
        if not (2 <= area <= 25 and width <= 8 and height <= 8):
            continue
        ys, xs = np.nonzero(labels[top:top + height, left:left + width] == i)
        px, py = int(left + xs[0]) + x, int(top + ys[0]) + y
        chips.append((px, py))
    return chips


def ore_supply(canvas, near, reach, region=None):
    """Count visible gold chips; a few regenerating crystals do not imply a full field."""
    return sum(abs(p[0] - near[0]) + abs(p[1] - near[1]) <= reach
               for p in _ore_chips(canvas, region))


def ore_target(canvas, near, reach, region=None):
    """A bright crystal pixel in a coherent ore patch, never the patch's bare-ground centroid."""
    chips = _ore_chips(canvas, region)
    candidates = [p for p in chips if
                  abs(p[0] - near[0]) + abs(p[1] - near[1]) <= reach and
                  sum(abs(p[0] - q[0]) < 65 and abs(p[1] - q[1]) < 35 for q in chips) >= 5]
    return min(candidates, key=lambda p: abs(p[0] - near[0]) + abs(p[1] - near[1])) if candidates else None


def gem_field(canvas, region=None):
    """Return a blue crystal pixel in a mixed-color field on visible ground."""
    if canvas is None:
        return None
    x, y, w, h = tactical(region)
    crop = canvas[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2HSV)
    b, g, r = (crop[..., i].astype(np.int16) for i in range(3))
    hue, saturation, value = (hsv[..., i] for i in range(3))
    masks = {
        'blue': (hue >= 115) & (hue <= 135) & (saturation >= 90) & (value >= 90) & (b - g >= 25),
        'red': ((hue <= 12) | (hue >= 170)) & (saturation >= 120) & (value >= 120) & (r - g >= 25),
        'green': (hue >= 45) & (hue <= 85) & (saturation >= 100) & (value >= 100) &
                 (g - r >= 15) & (g - b >= 15),
        'white': (saturation <= 60) & (value >= 195),
    }
    house = colors.OWN.mask(crop) | colors.enemy_mask(crop, compressed=True)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(house.astype(np.uint8), connectivity=8)
    occupied = np.zeros(house.shape, np.uint8)
    # Real gems include tiny red/blue glints; suppress substantial house-colored sprites only.
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            occupied[labels == i] = 1
    occupied = cv2.dilate(occupied, np.ones((31, 31), np.uint8))
    chips = {}
    for name, mask in masks.items():
        mask = (mask & (occupied == 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        chips[name] = []
        for i in range(1, count):
            left, top, width, height, area = stats[i]
            if 2 <= area <= 25 and width <= 8 and height <= 8:
                ys, xs = np.nonzero(labels[top:top + height, left:left + width] == i)
                chips[name].append((int(left + xs[0]), int(top + ys[0])))
    ground = ((hue >= 10) & (hue <= 95) & (saturation >= 30) & (saturation <= 180) &
              (value >= 50) & (value <= 230))
    fields = []
    minimum = {'blue': 6, 'red': 10, 'green': 15, 'white': 25}
    for px, py in chips['blue']:
        local = {name: sum(abs(px - qx) < 70 and abs(py - qy) < 40 for qx, qy in points)
                 for name, points in chips.items()}
        terrain = ground[max(0, py - 40):py + 40, max(0, px - 70):px + 70]
        if all(local[name] >= limit for name, limit in minimum.items()) and terrain.mean() > 0.65:
            fields.append(((px + x, py + y), sum(local.values())))
    return max(fields, key=lambda field: field[1]) if fields else None
