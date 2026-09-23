"""Production-tab glyphs and recorded blank buttons, independent of the visible cameo lane."""

from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def templates():
    with np.load(Path(__file__).parents[1] / 'profiles' / 'disabled-tabs-1024x768.npz') as data:
        return {key: data[key].astype(np.int16) for key in data.files}


def read_tabs(canvas, layout, blank=None):
    states = {}
    for lane, box in layout.tabs.items():
        states[lane] = None
        if (box.w, box.h) != (30, 32):
            continue
        pixels = box.crop(canvas)[5:-5, 6:-6].astype(np.int16)
        empty = (blank or templates())[lane]
        if np.abs(pixels - empty).mean() <= 8:
            states[lane] = False
        elif np.count_nonzero(pixels.max(axis=2) - pixels.min(axis=2) > 40) >= 30:
            states[lane] = True
    return states
