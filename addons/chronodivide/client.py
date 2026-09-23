"""Client profiles: how one RA2 client draws the game, so the inspector and the player work on any.

This is the Minesweeper desktop player's technique (addons/minesweeper/DESKTOP.md) applied to
RA2: the game is a window the user picks, its view a region found automatically or marked by
hand (calibration.py), and recognition a profile of samples taught from a captured frame, with
anything unknown rejected. Every RA2 client draws the same game art at 1:1 (Chrono Divide
needs the original game's files), so what differs between clients fits in a profile:

- where to look: window title hints and processes, and whether the game is drawn inside a page
  (a browser; the view is found on the page) or fills the window;
- the sidebar's geometry, from which the layout of any view size follows (Layout.derive);
- templates taught from frames: cameos, credits digits, HUD anchors, the cursor sprite;
- house colors (ours and the enemies'), and how the mouse reaches the game: 'locked' (browser
  pointer lock: relative motion, the game draws its own cursor) or 'absolute'.

Profiles are JSON files in profiles/clients/; template paths are relative to profiles/.
"""

import copy
import json
from pathlib import Path
import re

import numpy as np

from .layout import Layout, PROFILES
from .perception import colors
from .perception.colors import HouseColor
from .perception.cursor import CursorFinder
from .perception.hud import CreditsReader
from .perception.modes import ModeReader
from .perception.sidebar import CameoCatalog

CLIENTS = PROFILES / 'clients'
DEFAULT = 'chronodivide'
TEMPLATES = ('cameos', 'digits', 'hud', 'cursor')


def slug(name):
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'client'


class ClientProfile:
    def __init__(self, raw, path=None):
        if raw.get('version') != 1:
            raise ValueError('Unsupported client profile version.')
        self.raw, self.path = raw, Path(path) if path else None
        self.name = str(raw['name'])
        window = raw.get('window', {})
        self.titles = tuple(window.get('titles', ()))
        self.processes = tuple(p.lower() for p in window.get('processes', ()))
        self.embedded = bool(window.get('embedded', False))
        self.input = raw.get('input', 'absolute')
        if self.input not in ('locked', 'absolute'):
            raise ValueError(f'Unknown input mode {self.input!r}: use "locked" or "absolute".')
        view = raw.get('view', {})
        self.size = tuple(view.get('size', (1024, 768)))
        self.bottom_bar, self.command_bar = int(view.get('bottom_bar', 32)), int(view.get('command_bar', 30))
        self.sidebar = raw['sidebar']
        self.templates = dict(raw.get('templates', {}))
        house = raw.get('colors', {})
        self.own = HouseColor.from_json(house.get('own', 'red'))
        self.enemies = tuple(HouseColor.from_json(c) for c in house.get('enemies', ['blue']))
        self.allies = tuple(HouseColor.from_json(c) for c in house.get('allies', []))
        self._check_colors(self.own, self.enemies, self.allies)
        self.verified = raw.get('verified', '')
        self.detector = raw.get('detector')

    @property
    def slug(self):
        return self.path.stem if self.path else slug(self.name)

    @classmethod
    def load(cls, name=DEFAULT):
        """A profile by name (profiles/clients/<name>.json) or by path."""
        path = Path(name)
        if path.suffix.lower() != '.json':
            path = CLIENTS / f'{name}.json'
        return cls(json.loads(path.read_text(encoding='utf-8')), path)

    @staticmethod
    def available(folder=CLIENTS):
        return sorted(p.stem for p in Path(folder).glob('*.json'))

    # -- geometry and readers ---------------------------------------------------
    def layout(self, width=None, height=None):
        width, height = int(width or self.size[0]), int(height or self.size[1])
        return Layout.derive(self.sidebar, width, height, self.bottom_bar, self.command_bar,
                             name=f'{self.slug}-{width}x{height}')

    def template_path(self, kind):
        relative = self.templates.get(kind)
        return None if not relative else PROFILES / relative

    def _existing(self, kind):
        path = self.template_path(kind)
        return path if path is not None and path.exists() else None

    def cameos(self):
        path = self._existing('cameos')
        options = self.raw.get('cameo_style', {})
        return CameoCatalog.load(path, **options) if path else CameoCatalog({}, **options)

    def credits(self):
        path = self._existing('digits')
        options = self.raw.get('credits_style', {})
        return CreditsReader.load(path, **options) if path else CreditsReader(**options)

    def modes(self, layout):
        anchors = {name: box.tuple() for name, box in layout.anchors.items()}
        path = self._existing('hud')
        if path is None:
            return ModeReader({}, anchors=anchors, menu=layout.menu.tuple())
        return ModeReader.load(path, anchors=anchors, menu=layout.menu.tuple())

    def cursor(self, layout):
        path = self._existing('cursor')
        return CursorFinder.load(path, radar=layout.regions['radar'].tuple()) if path else None

    def hud_templates(self):
        """The HUD anchor crops (name -> BGR uint8), for finding the view in a window."""
        path = self._existing('hud')
        if path is None:
            return {}
        with np.load(path) as data:
            return {name: data[name].astype(np.uint8) for name in data.files}

    def use_colors(self):
        """Make this profile's house colors the ones every reader uses; returns the previous."""
        return colors.use(self.own, self.enemies, self.allies)

    # -- teaching -----------------------------------------------------------------
    def derived(self, name):
        """A new profile named `name` with this one's settings (saved to profiles/clients/)."""
        raw = copy.deepcopy(self.raw)
        raw['name'] = name
        raw['verified'] = ''
        raw['about'] = f'Taught from "{self.name}". ' + raw.get('about', '')
        return ClientProfile(raw, CLIENTS / f'{slug(name)}.json')

    @staticmethod
    def _check_colors(own, enemies, allies):
        names = [c.name for c in (own, *enemies, *allies)]
        if len(names) != len(set(names)):
            raise ValueError('Each house color must belong to exactly one side: ours, enemy or ally.')

    def set_colors(self, own=None, enemies=None, allies=None):
        own = own or self.own
        enemies = tuple(enemies) if enemies is not None else self.enemies
        allies = tuple(allies) if allies is not None else self.allies
        self._check_colors(own, enemies, allies)
        self.own, self.enemies, self.allies = own, enemies, allies
        self.raw['colors'] = dict(own=own.to_json(), enemies=[c.to_json() for c in enemies],
                                  allies=[c.to_json() for c in allies])

    def save(self, catalog=None, credits=None, modes=None):
        """Write the profile, and any taught readers into a folder beside it named after it
        (profiles/clients/<slug>/): a profile never writes over another profile's templates."""
        path = (self.path or CLIENTS / f'{self.slug}.json').resolve()
        own = path.parent / path.stem
        for kind, reader in (('cameos', catalog), ('digits', credits), ('hud', modes)):
            if reader is None:
                continue
            own.mkdir(parents=True, exist_ok=True)
            target = own / f'{kind}.npz'
            reader.save(target)
            try:
                self.templates[kind] = target.relative_to(PROFILES).as_posix()
            except ValueError:
                self.templates[kind] = str(target)             # a profile kept elsewhere
        self.raw['templates'] = dict(self.templates)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.raw, indent=2) + '\n', encoding='utf-8')
        self.path = path
        return path
