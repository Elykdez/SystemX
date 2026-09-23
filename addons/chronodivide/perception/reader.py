"""All per-frame readers bundled: one call turns a canvas into world updates."""

from ..world import Slot
from .hud import read_power
from .radar import read_radar
from .sidebar import lane_of, read_slots
from .tabs import read_tabs


class Perception:
    """The readers of one client profile (client.py) for one view size; Chrono Divide at 1024x768
    unless told otherwise."""

    def __init__(self, layout=None, catalog=None, credits=None, modes=None, cursor=None, profile=None,
                 compressed=False):
        from ..client import ClientProfile
        self.profile = profile or ClientProfile.load()
        self.layout = layout or self.profile.layout()
        self.catalog = catalog or self.profile.cameos()
        self.credits = credits or self.profile.credits()
        self.modes = modes or self.profile.modes(self.layout)
        self.cursor = cursor or self.profile.cursor(self.layout)
        self.blank_tabs = None
        tab_path = self.profile.template_path('disabled_tabs')
        if tab_path is not None:
            import numpy as np
            with np.load(tab_path) as data:
                self.blank_tabs = {key: data[key].astype(np.int16) for key in data.files}
        self.compressed = compressed
        self.credits.compressed = self.catalog.compressed = compressed
        self.modes.compressed = compressed

    def use_layout(self, layout):
        """Read a view of another size: the same templates in the new layout's places."""
        self.layout = layout
        self.modes.anchors = {name: box.tuple() for name, box in layout.anchors.items()}
        self.modes.menu = layout.menu.tuple()
        if self.cursor is not None:
            self.cursor.radar = layout.regions['radar'].tuple()

    def read(self, canvas, world, now=None):
        """Update `world` from one canvas; returns (mode dict, slot readings, visible lane)."""
        mode = self.modes.read(canvas)
        world.set('mode', mode['mode'], now)
        readings, lane = [], None
        if mode['mode'] == 'hud':
            value = self.credits.read(self.layout.regions['credits'].crop(canvas))
            if value is not None:
                world.set('credits', value, now)
            world.set('power', read_power(self.layout.regions['power'].crop(canvas)), now)
            world.set('radar', read_radar(canvas, self.layout.regions['radar'], self.compressed,
                                         self.profile.raw.get('radar_style', {}).get('line_color', 'yellow')), now)
            readings = read_slots(canvas, self.layout, self.catalog)
            # A finished unit can show Ready at a blocked exit without a queue numeral.
            # The catalog verifies the Ready text independently of that numeral.
            # Only before the MCV deploys is the whole sidebar empty; deployment depends on it.
            world.set('sidebar_empty', all(r.state == 'empty' for r in readings), now)
            lane = lane_of(readings, world.knowledge)
            if lane:
                world.update_sidebar(lane, [Slot(i, r.item, r.state, r.progress, 1 if r.queued else None)
                                            for i, r in enumerate(readings)], now)
        # Absence is evidence only when every slot was read; partial views cannot prove a loss.
        slots = [Slot(i, r.item, r.state, r.progress, 1 if r.queued else None)
                 for i, r in enumerate(readings)]
        complete = len(readings) == len(self.layout.grid.boxes()) and all(
            r.state not in ('unknown', 'menu') for r in readings)
        world.reconcile_structures(lane, slots, complete=complete, now=now)
        world.reconcile_tabs(read_tabs(canvas, self.layout, self.blank_tabs) if mode['mode'] == 'hud' else {}, now=now)
        return mode, readings, lane
