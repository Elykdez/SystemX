"""Conservative appearance recognition for recorded native building sprites.

The templates compare grayscale facade/crane detail, not house-color components. Each match
anchors one full sprite box; house color supplies ownership only after appearance matches.
These are fixed-scale templates for the native game sprites, including damaged yard and factory appearances.
Other damage states, occlusion and scaled clients may be missed; no generic building count is implied.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import side_of
from .view import tactical

PROFILE = Path(__file__).resolve().parents[1] / 'profiles' / 'buildings-1024x768.npz'


@dataclass(frozen=True)
class Building:
    box: tuple
    side: str
    name: str
    score: float
    kind: str = 'building'
    source: str = 'template'
    sure: bool = True


class BuildingRecognizer:
    def __init__(self, path=PROFILE):
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        with np.load(path, allow_pickle=False) as data:
            self.templates = [(row, data[row.get('template', row['name'])].copy()) for row in meta['templates']]

    def detect(self, canvas, region=None, compressed=False):
        """Recognized buildings in an (x, y, w, h) region; boxes use canvas coordinates."""
        x, y, w, h = tactical(region)
        crop = canvas[y:y + h, x:x + w]
        if crop.size == 0:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        found = []
        for row, template in self.templates:
            th, tw = template.shape
            if gray.shape[0] < th or gray.shape[1] < tw:
                continue
            response = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            # Neighboring correlation peaks describe the same sprite, not extra buildings.
            while True:
                _, score, _, (mx, my) = cv2.minMaxLoc(response)
                if score < row['threshold']:
                    break
                response[max(0, my - th // 2):my + th // 2 + 1,
                         max(0, mx - tw // 2):mx + tw // 2 + 1] = -1
                left, top, right, bottom = row['box_offset']
                box = (max(x, x + mx + left), max(y, y + my + top),
                       min(x + crop.shape[1], x + mx + right),
                       min(y + crop.shape[0], y + my + bottom))
                side = side_of(canvas, box, share=0.01, compressed=compressed)
                if side is not None:
                    building = Building(box, side, row['name'], round(score, 3), sure=score >= 0.8)
                    duplicate = next((b for b in found if b.name == building.name and b.side == side
                                      and abs(b.box[0]-box[0])+abs(b.box[1]-box[1]) < 40), None)
                    if duplicate is None:
                        found.append(building)
                    elif building.score > duplicate.score:
                        found[found.index(duplicate)] = building
        return found


@lru_cache(maxsize=1)
def default_recognizer():
    """Load the small bundled templates once; scans keep no mutable detector state."""
    return BuildingRecognizer()
