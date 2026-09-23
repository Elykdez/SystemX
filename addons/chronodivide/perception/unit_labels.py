"""Read native hover names near a selected unit, rather than guessing its type."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=1)
def templates():
    with np.load(Path(__file__).resolve().parents[1] / 'profiles' / 'unit-labels-allied-1024x768.npz') as data:
        return {name: data[name] for name in data.files}


def read_label(canvas, point):
    if canvas is None:
        return None
    x, y = point
    # Native labels appear 20 pixels to either side/below the hovered ground tile.
    x0, y0 = max(0, x - 150), max(0, y - 55)
    crop = canvas[y0:min(canvas.shape[0], y + 60), x0:min(canvas.shape[1], x + 150)]
    if crop.size == 0:
        return None
    ink = (crop[:, :, 0] > 60).astype(np.uint8)
    scores = []
    for name, template in templates().items():
        # The Soviet yellow tooltip loses strokes in the blue channel.
        label_ink = ((crop[:, :, 1] > 160) & (crop[:, :, 2] > 160)).astype(np.uint8) if name == 'war_miner' else ink
        if any(a < b for a, b in zip(label_ink.shape, template.shape)):
            continue
        score = cv2.minMaxLoc(cv2.matchTemplate(label_ink, template, cv2.TM_CCOEFF_NORMED))[1]
        scores.append((score, name))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 0.85 or (len(scores) > 1 and scores[0][0] - scores[1][0] < 0.08):
        return None
    return scores[0][1]
