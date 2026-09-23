"""HUD layout in canvas pixels: loaded from a calibrated file for one canvas size, or derived
for any size from a client profile's sidebar geometry (`Layout.derive`)."""

from dataclasses import dataclass
import json
from pathlib import Path

from desktop.target import DesktopPause

PROFILES = Path(__file__).with_name('profiles')
ANCHORS = {'top_buttons': (874, 20, 132, 24), 'sidebar_bottom': (850, 736, 174, 32)}   # 1024x768
MENU = (877, 228, 128, 200)


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def crop(self, image):
        if self.x < 0 or self.y < 0 or self.x + self.w > image.shape[1] or self.y + self.h > image.shape[0]:
            raise ValueError(f'{self} is outside the {image.shape[1]}x{image.shape[0]} frame.')
        return image[self.y:self.y + self.h, self.x:self.x + self.w]

    def contains(self, point):
        return self.x <= point[0] < self.x + self.w and self.y <= point[1] < self.y + self.h

    def tuple(self):
        return self.x, self.y, self.w, self.h


@dataclass(frozen=True)
class SlotGrid:
    x: int
    y: int
    cols: int
    rows: int
    width: int
    height: int
    pitch_x: int
    pitch_y: int

    def box(self, index):
        row, col = divmod(index, self.cols)
        return Box(self.x + col * self.pitch_x, self.y + row * self.pitch_y, self.width, self.height)

    def boxes(self):
        return [self.box(i) for i in range(self.cols * self.rows)]


class Layout:
    def __init__(self, raw):
        if raw.get('version') != 1:
            raise ValueError('Unsupported layout version.')
        self.raw = raw
        self.name = raw.get('name', 'layout')
        self.calibrated = bool(raw.get('calibrated'))
        self.canvas = tuple(raw['canvas'])
        self.regions = {k: Box(*v) for k, v in raw['regions'].items()}
        self.tabs = {k: Box(*v) for k, v in raw['tabs'].items()}
        self.grid = SlotGrid(**raw['grid'])
        self.points = {k: tuple(v) for k, v in raw.get('points', {}).items()}
        # HUD anchors (the in-match sidebar, see perception/modes.py) and where the game menu's red
        # buttons appear; the measured 1024x768 places when a file does not give them.
        self.anchors = {k: Box(*v) for k, v in raw.get('anchors', ANCHORS).items()}
        self.menu = Box(*raw.get('menu', MENU))
        boxes = [*self.regions.items(), *self.tabs.items(), *self.anchors.items(), ('menu', self.menu),
                 *((f'slot{i}', b) for i, b in enumerate(self.grid.boxes()))]
        for name, box in boxes:
            if box.x < 0 or box.y < 0 or box.x + box.w > self.canvas[0] or box.y + box.h > self.canvas[1]:
                raise ValueError(f'Layout region {name} lies outside the {self.canvas} canvas.')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def derive(cls, sidebar, width, height, bottom_bar=32, command_bar=30, name=None):
        """The layout of a width x height game view, from a client profile's sidebar geometry.

        RA2 draws a fixed-width sidebar at the view's right edge. Its parts sit at fixed offsets
        from its top-left corner, except those anchored to the bottom (a negative y counts from
        the bottom; a height of zero or less stretches the box down to that far above the
        bottom), and the cameo grid shows as many rows as fit above its bottom margin. The
        tactical view is everything left of the sidebar, above the command bar. At 1024x768
        this gives the measured layout exactly (tests/test_client.py).
        """
        left = width - sidebar['width']

        def box(spec, x0=left):
            x, y, w, h = spec
            top = y if y >= 0 else height + y
            return [x0 + x, top, w, h if h > 0 else height + h - top]

        def point(spec):
            x, y = spec
            return [left + x, y if y >= 0 else height + y]
        grid = dict(sidebar['grid'])
        margin = grid.pop('bottom')
        grid.update(x=left + grid['x'], rows=max(1, (height - grid['y'] - margin) // grid['pitch_y']))
        tactical_h = height - bottom_bar
        return cls(dict(
            version=1, name=name or f'derived-{width}x{height}', calibrated=False, canvas=[width, height],
            regions={'tactical': [0, 0, left, tactical_h], 'command_bar': [0, height - command_bar, left, command_bar],
                     'sidebar': [left, 0, sidebar['width'], height],
                     **{k: box(v) for k, v in sidebar['regions'].items()}},
            tabs={k: box(v) for k, v in sidebar['tabs'].items()},
            grid=grid,
            anchors={k: box(v) for k, v in sidebar['anchors'].items()},
            menu=box(sidebar['menu']),
            points={**{k: point(v) for k, v in sidebar.get('points', {}).items()
                       if k != 'waypoint' or (width, height) == (1024, 768)},
                    # Where the agent clicks in the tactical view: a right click on open ground clears
                    # the selection; the MCV and then the construction yard stand near the center.
                    'recapture': [left // 2, height - 118], 'tactical_center': [left // 2, tactical_h // 2],
                    'home': [left // 2 + 3, tactical_h // 2 - 8]}))

    @classmethod
    def for_canvas(cls, width, height, folder=PROFILES):
        path = Path(folder) / f'layout-{width}x{height}.json'
        if not path.exists():
            raise DesktopPause(f'No HUD layout for a {width}x{height} game canvas. Use the calibrated size '
                               f'or calibrate a new layout.')
        return cls.load(path)

    def check(self, canvas):
        if (canvas.width, canvas.height) != self.canvas:
            raise DesktopPause(f'Game canvas is {canvas.width}x{canvas.height}; layout {self.name} expects '
                               f'{self.canvas[0]}x{self.canvas[1]}.')

    @property
    def tactical(self):
        return self.regions['tactical']

    @property
    def sidebar(self):
        return self.regions.get('sidebar') or Box(self.tactical.w, 0, self.canvas[0] - self.tactical.w, self.canvas[1])
