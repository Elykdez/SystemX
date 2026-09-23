"""Autoplay between matches: Continue on the score screen, Start Game on the skirmish setup.

With `--autoplay` the controller plays one match, exits when the score screen appears,
and `play-autoplay.bat` starts it again, so every match runs the latest code. In menus
the player asks the pilot what to click: a known screen must stay the same for a moment
before its button is pressed, presses are spaced out, and nothing is pressed right after
physical mouse or keyboard use. Screens are recognized by their header (tools/menu.py).
Unknown screens are left alone.
"""

import time

from .tools.menu import SCREENS, identify, load


class MenuPilot:
    def __init__(self, templates=None, settle=2.0, gap=6.0, quiet=10.0, max_clicks=12):
        self.templates = load() if templates is None else templates
        self.settle, self.gap, self.quiet, self.max_clicks = settle, gap, quiet, max_clicks
        self.seen, self.since = None, 0.0
        self.last_click = -1e9
        self.clicks = 0

    def step(self, canvas, last_physical=None, now=None):
        """(screen, canvas point) to click now, or None."""
        now = time.monotonic() if now is None else now
        screen = identify(canvas, self.templates) if canvas is not None else None
        if screen != self.seen:
            self.seen, self.since = screen, now
            return None
        if screen is None or now - self.since < self.settle or now - self.last_click < self.gap:
            return None
        if last_physical is not None and now - last_physical < self.quiet:
            return None
        if self.clicks >= self.max_clicks:
            return None                    # something is wrong with this screen: stop pressing
        self.last_click = now
        self.clicks += 1
        return screen, SCREENS[screen][1]
