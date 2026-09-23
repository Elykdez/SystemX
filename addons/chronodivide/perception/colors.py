"""House colors: which pixels belong to us, and which to the enemy.

RA2 draws every side's units and buildings from the same sprites and remaps one range of
palette entries to the side's house color: a ramp of one hue, dark to light. Two colors were
measured on Chrono Divide (2026-09-23..27) as exact channel-ratio tests, in the tactical view
and on the radar: Russia's red and the opponent's blue. Any other color is taught by clicking
a unit or building of that side in the inspector's setup; it is kept as a hue with a
tolerance and minimum saturation and brightness. That test is looser than the measured ones:
ore, gems and water can share a hue with a house color.

The player sets the colors of the match from its client profile (`use`); readers that are not
handed colors use ours (OWN) and the enemies' (ENEMIES).
"""

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


def _channels(bgr, dtype):
    return (bgr[..., i].astype(dtype) for i in range(3))


def _red(view):
    b, g, r = _channels(view, np.int16)
    return (r > 140) & (g < 80) & (b < 80) & (r - g > 90)


def _red_radar(crop):
    b, g, r = _channels(crop, np.int16)
    return (r > 150) & (g < 90) & (b < 90) & (r - g > 100)


def _blue(view):
    """G about 0.49 B and R about 0.16 B, from (114, 56, 18) to (224, 114, 42). Our Tesla Reactors
    light up in flat (220, 70, 70) and (147, 0, 0), which fail the ratio test."""
    b, g, r = _channels(view, np.int32)
    return (b > 100) & (100 * g >= 42 * b) & (100 * g <= 56 * b) & (100 * r >= 10 * b) & (100 * r <= 25 * b)


def _blue_radar(crop):
    b, g, r = _channels(crop, np.int32)
    return (b >= 140) & (100 * g >= 40 * b) & (100 * g <= 58 * b) & (100 * r >= 8 * b) & (100 * r <= 26 * b)


def _blue_compressed(view):
    """JPEG mixes palette blue with its background; hue survives better than channel ratios.

    The measured enemy ramp lies near hue 108, while reactor glow lies near 120. Keep that
    separation and require chroma, brightness and green above red, rather than accepting blue.
    """
    hsv = cv2.cvtColor(np.ascontiguousarray(view, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    b, g, r = _channels(view, np.int16)
    mask = (hsv[..., 0] >= 105) & (hsv[..., 0] <= 112) & (hsv[..., 1] >= 100) & (b >= 90) & (g - r >= 12)
    # Compression mixes violet reactor glow with green ground into the enemy's hue. Reject
    # fragments around a substantial glow region; JPEG ringing extends beyond its bounds.
    glow = ((hsv[..., 0] >= 116) & (hsv[..., 0] <= 124) &
            (hsv[..., 1] >= 100) & (hsv[..., 2] >= 110)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(glow, connectivity=8)
    for x, y, width, height, area in stats[1:]:
        if area >= 50:
            mask[max(0, y - 8):y + height + 8, max(0, x - 8):x + width + 8] = False
    return mask


MEASURED = {'red': (_red, _red_radar), 'blue': (_blue, _blue_radar)}


@dataclass(frozen=True)
class HouseColor:
    name: str
    hue: float | None = None          # OpenCV hue, 0..180; None: a measured color (MEASURED)
    tolerance: float = 8.0            # hue units either side
    saturation: int = 110             # minimum, 0..255
    value: int = 60                   # minimum brightness, 0..255

    def mask(self, bgr, compressed=False):
        """Pixels of this color in a BGR image (the tactical view or a part of it)."""
        if self.hue is None:
            if compressed and self.name == 'blue':
                return _blue_compressed(bgr)
            return MEASURED[self.name][0](bgr)
        return self._hue_mask(bgr)

    def radar_mask(self, bgr):
        """Dots of this color on the radar (drawn flat, in the ramp's brightest shade)."""
        if self.hue is None:
            return MEASURED[self.name][1](bgr)
        return self._hue_mask(bgr, value=max(self.value, 120))

    def _hue_mask(self, bgr, value=None):
        hsv = cv2.cvtColor(np.ascontiguousarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2HSV)
        distance = np.abs(hsv[..., 0].astype(np.int16) - int(round(self.hue)))
        distance = np.minimum(distance, 180 - distance)
        return (distance <= self.tolerance) & (hsv[..., 1] >= self.saturation) & \
            (hsv[..., 2] >= (self.value if value is None else value))

    def to_json(self):
        return {k: v for k, v in asdict(self).items() if v is not None and (self.hue is not None or k == 'name')}

    @classmethod
    def from_json(cls, raw):
        if isinstance(raw, str):
            raw = {'name': raw}
        color = cls(str(raw['name']), raw.get('hue'), float(raw.get('tolerance', 8.0)),
                    int(raw.get('saturation', 110)), int(raw.get('value', 60)))
        if color.hue is None and color.name not in MEASURED:
            raise ValueError(f'House color {color.name!r} has no hue and is not a measured color '
                             f'({", ".join(MEASURED)}). Teach it in the setup window.')
        return color

    @classmethod
    def sample(cls, name, patch):
        """A color taught from pixels around a click on a unit or building of that side."""
        hsv = cv2.cvtColor(np.ascontiguousarray(patch, dtype=np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        vivid = hsv[(hsv[:, 1] >= 110) & (hsv[:, 2] >= 60)].astype(np.float64)
        if len(vivid) < 12:
            raise ValueError('No house color there: click on a unit or building of that side.')
        angles = vivid[:, 0] * (2 * math.pi / 180)
        hue = (math.degrees(math.atan2(np.sin(angles).mean(), np.cos(angles).mean())) / 2) % 180
        distance = np.abs(vivid[:, 0] - hue)
        distance = np.minimum(distance, 180 - distance)
        return cls(name, round(hue, 1), float(np.clip(np.percentile(distance, 90) + 2, 5, 14)),
                   int(max(90, np.percentile(vivid[:, 1], 10) - 10)), int(max(50, np.percentile(vivid[:, 2], 10) - 10)))


RED, BLUE = HouseColor('red'), HouseColor('blue')
OWN = RED
ENEMIES = (BLUE,)
ALLIES = ()


def use(own, enemies, allies=()):
    """Set match relationships; returns the previous colors for restoration."""
    global OWN, ENEMIES, ALLIES
    previous = OWN, ENEMIES, ALLIES
    OWN, ENEMIES, ALLIES = own, tuple(enemies), tuple(allies)
    return previous


def enemy_mask(bgr, compressed=False):
    mask = np.zeros(bgr.shape[:2], bool)
    for color in ENEMIES:
        mask |= color.mask(bgr, compressed=compressed)
    return mask & ~ally_mask(bgr, compressed=compressed)


def ally_mask(bgr, compressed=False):
    mask = np.zeros(bgr.shape[:2], bool)
    for color in ALLIES:
        mask |= color.mask(bgr, compressed=compressed)
    return mask


def enemy_radar_mask(bgr):
    mask = np.zeros(bgr.shape[:2], bool)
    for color in ENEMIES:
        mask |= color.radar_mask(bgr)
    for color in ALLIES:
        mask &= ~color.radar_mask(bgr)
    return mask
