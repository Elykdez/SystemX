"""Recognize the recorded deployed GI's body and sandbags, independently of building color.

Three native-size facing/animation appearances from run 20260929-105629-188247,
frame-1790650649225476200: crops (518,390,547,421), (577,422,608,452),
(607,285,638,318). Unknown infantry,
other scales and occluded sprites are not classified. These are ground targets;
airborne Rocketeers are deliberately not treated as deployed GIs.

The two east-firing poses are from the 2026-10-08 browser trial, canvas crop
[444:474,54:84]. A later frame supplies their held-out regression image.
Three southeast-firing poses come from the 2026-10-09 Korea trial,
frame-1791525248876017900, crops (396,330,31,33), (395,363,31,31), (488,341,31,35).
The subsequent frame-1791525249940643500 is the held-out Allied-base fixture.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import colors, view

PROFILE = Path(__file__).resolve().parents[1] / 'profiles' / 'deployed-gi-1024x768.npz'


@lru_cache(maxsize=1)
def _templates():
    with np.load(PROFILE, allow_pickle=False) as data:
        return tuple(data[key].copy() for key in data.files)


def deployed_gi_targets(canvas, region=None, compressed=False):
    """Canvas points on template-confirmed enemy deployed GIs, suitable for ground fighting."""
    x, y, w, h = view.tactical(region)
    crop = canvas[y:y + h, x:x + w]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    found = []
    for template in _templates():
        th, tw = template.shape[:2]
        reference = uniform = None
        if template.ndim == 3:
            blue = np.array([[[224, 114, 42]]], np.uint8)
            if not colors.enemy_mask(blue).any():
                continue
            reference = template.astype(np.int16)
            uniform = ((reference[..., 0] > reference[..., 2] + 10)
                       & (reference[..., 0] > reference[..., 1] + 5))
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template
        if gray.shape[0] < th or gray.shape[1] < tw:
            continue
        response = cv2.matchTemplate(gray, gray_template, cv2.TM_CCOEFF_NORMED)
        while True:
            _, score, _, (mx, my) = cv2.minMaxLoc(response)
            if score < 0.84:
                break
            response[max(0, my - th // 2):my + th // 2 + 1,
                     max(0, mx - tw // 2):mx + tw // 2 + 1] = -1
            sprite = crop[my:my + th, mx:mx + tw]
            if reference is not None:
                # JPEG can erase strict blue pixels; require the recorded body and uniform instead.
                if cv2.matchTemplate(sprite, template, cv2.TM_CCOEFF_NORMED)[0, 0] < 0.82:
                    continue
                sprite = sprite.astype(np.int16)
                if (np.abs(sprite - reference)[uniform].mean() > 26
                        or (sprite[..., 0] - sprite[..., 2])[uniform].mean() < 10):
                    continue
            elif colors.enemy_mask(sprite, compressed=compressed).sum() < 3:
                continue
            point = (x + mx + tw // 2, y + my + th // 2)
            if not any((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 < 18 ** 2 for p in found):
                found.append(point)
    return found


@lru_cache(maxsize=1)
def _standing_templates():
    with np.load(PROFILE.with_name('standing-gi-1024x768.npz'), allow_pickle=False) as data:
        templates = []
        for key in data.files:
            template = data[key].copy()
            reference = template.astype(np.int16)
            uniform = ((reference[..., 0] > reference[..., 2] + 10)
                       & (reference[..., 0] > reference[..., 1] + 5))
            templates.append((template, cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), reference, uniform))
        return tuple(templates)


def standing_gi_targets(canvas, region=None, compressed=False):
    """Native-size blue GI standing/firing poses, matched by body appearance and uniform color.

    Sources: standing-gi-raid-browser.png [451:476,494:510] and speed-6 run
    20260929-163758-864944/frame-1790671531432576600 [450:475,699:718]. The latter
    pose can have zero pixels in the strict JPEG house mask; a full-color match plus
    agreement at the reference uniform pixels supplies the evidence instead. This profile
    supports blue enemies only and abstains on other palettes, scales or unseen poses.
    """
    blue = np.array([[[224, 114, 42]]], np.uint8)
    if not colors.enemy_mask(blue).any():
        return []
    x, y, w, h = view.tactical(region)
    crop = canvas[y:y + h, x:x + w]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    found = []
    for template, gray_template, reference, uniform in _standing_templates():
        th, tw = template.shape[:2]
        if crop.shape[0] < th or crop.shape[1] < tw:
            continue
        # Search cheaply in grayscale; retain the original full-color check at each candidate.
        response = cv2.matchTemplate(gray, gray_template, cv2.TM_CCOEFF_NORMED)
        while True:
            _, score, _, (mx, my) = cv2.minMaxLoc(response)
            if score < 0.78:
                break
            response[max(0, my - th // 2):my + th // 2 + 1,
                     max(0, mx - tw // 2):mx + tw // 2 + 1] = -1
            sprite = crop[my:my + th, mx:mx + tw]
            if cv2.matchTemplate(sprite, template, cv2.TM_CCOEFF_NORMED)[0, 0] < 0.82:
                continue
            sprite = sprite.astype(np.int16)
            if (np.abs(sprite - reference)[uniform].mean() > 26
                    or (sprite[..., 0] - sprite[..., 2])[uniform].mean() < 10):
                continue
            point = (x + mx + tw // 2, y + my + th // 2)
            if not any((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 < 12 ** 2 for p in found):
                found.append(point)
    return found
