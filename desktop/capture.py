"""Screen capture in physical pixels; capture handles belong to their worker thread."""

from dataclasses import dataclass
import time
import numpy as np
from mss import MSS

from .target import DesktopPause, Geometry


@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp: float
    geometry: Geometry
    pixels: np.ndarray
    origin: tuple[int, int] = (0, 0)


class Capture:
    def __init__(self, target):
        self.target = target
        self.sequence = 0
        self.source = MSS()

    def read(self, region=None):
        geometry = self.target.geometry()
        g = geometry
        self.target.check_point(g.x + g.width / 2, g.y + g.height / 2, foreground=True)
        screen = self.source.monitors[0]
        if (g.x < screen['left'] or g.y < screen['top'] or
                g.x + g.width > screen['left'] + screen['width'] or
                g.y + g.height > screen['top'] + screen['height']):
            raise DesktopPause("Keep the complete game window on the desktop.")
        x, y, width, height = region if region is not None else (0, 0, g.width, g.height)
        if (any(type(v) is not int for v in (x, y, width, height)) or
                min(x, y) < 0 or min(width, height) <= 0 or
                x + width > g.width or y + height > g.height):
            raise DesktopPause('Capture region is outside the game window. Recalibrate.')
        pixels = np.asarray(self.source.grab(dict(left=g.x + x, top=g.y + y, width=width, height=height)))[:, :, :3].copy()
        if self.target.geometry() != geometry:
            raise DesktopPause("Window moved during capture.")
        self.sequence += 1
        return Frame(self.sequence, time.monotonic(), geometry, pixels, (x, y))

    def close(self):
        self.source.close()
