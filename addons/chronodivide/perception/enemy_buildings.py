"""Native sprite evidence for a damaged red Tesla Coil with sparse house trim."""

from functools import lru_cache
from pathlib import Path

import cv2

from . import colors, view


@lru_cache(maxsize=1)
def _tesla_base():
    return cv2.imread(str(Path(__file__).resolve().parents[1] / 'profiles' /
                          'damaged-tesla-base-1024x768.png'))


def tesla_coil_targets(canvas, region=None, compressed=False):
    # Recorded from 20261017-092314-allied/west-base-assault.png, canvas
    # (218,264,35,25). Match the surviving base, excluding animated fire above it.
    x, y, w, h = view.tactical(region)
    crop, template = canvas[y:y + h, x:x + w], _tesla_base()
    th, tw = template.shape[:2]
    if crop.shape[0] < th or crop.shape[1] < tw:
        return []
    response = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    found = []
    while True:
        _, score, _, (mx, my) = cv2.minMaxLoc(response)
        if score < 0.9:
            return found
        response[max(0, my - th // 2):my + th // 2 + 1,
                 max(0, mx - tw // 2):mx + tw // 2 + 1] = -1
        if colors.enemy_mask(crop[my:my + th, mx:mx + tw], compressed=compressed).sum() >= 5:
            found.append((x + mx + tw // 2, y + my + th - 1))
