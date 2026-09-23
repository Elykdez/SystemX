"""Test operator: click through the game's menus with OS input. Not part of the agent.

The agent (System One with its reflexes and skills) plays only once a match is being played;
getting there is the tester's job. For unattended test runs this tool does it:

    python -m addons.chronodivide.tools.menu screen      which menu screen is showing
    python -m addons.chronodivide.tools.menu click X Y   hover, then click a canvas point
    python -m addons.chronodivide.tools.menu start       score screen: Continue; setup: Start Game;
                                                         main menu: Demo Mode (skirmish against the AI)

Menu buttons need the pointer to rest on them before a click counts, so every click is a
move, a short pause, then the button press. Screens are recognized by their right-panel
header, learned from recorded frames (profiles/screens-1024x768.npz).
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from desktop import input as di
from desktop.target import DesktopPause, enable_dpi
from ..calibration import open_view
from ..window import bring_to_front

PROFILE = Path(__file__).resolve().parents[1] / 'profiles' / 'screens-1024x768.npz'
# screen -> header box (x, y, w, h) in canvas pixels and its main button
SCREENS = {'donation': ((592, 503, 126, 24), (654, 514)),  # donation request after a match: Maybe Later
                                                           # (the audio notice after a browser start has
                                                           # its OK there too and matches this template)
           'setup': ((758, 80, 142, 32), (828, 346)),     # "Skirmish Game" header, Start Game
           'main': ((758, 86, 142, 32), (828, 388)),      # "Main Menu" header, Demo Mode (skirmish vs AI)
           'options': ((758, 86, 142, 32), (828, 640)),   # "Options" header, Back
           'score': ((758, 86, 142, 32), (829, 640))}     # "Skirmish Score" header, Continue
MAX_DIFFERENCE = 14.0
# The main menu's Fullscreen button is where Continue is on the score screen, and its header
# differs from "Skirmish Score" by only 10.5 (score screens: 4.2): the score needs a close match.
LIMITS = {'score': 8.0}


def crop(canvas, box):
    x, y, w, h = box
    return canvas[y:y + h, x:x + w].astype(np.int16)


def learn(frames):
    """frames: screen -> canvas image; saves the header templates."""
    np.savez_compressed(PROFILE, **{name: crop(image, SCREENS[name][0]).astype(np.uint8)
                                    for name, image in frames.items()})


def load():
    with np.load(PROFILE) as data:
        return {name: data[name].astype(np.int16) for name in data.files}


def identify(canvas, templates):
    """Name of the menu screen showing, or None. Dialogs are checked first: the donation
    request is drawn over the score screen, whose header stays visible. Among headers in the
    same place, the closest match wins. The headers were learned on a 1024x768 view only."""
    if canvas.shape[:2] != (768, 1024):
        return None
    best = None
    for name in SCREENS:
        template = templates.get(name)
        if template is None:
            continue
        difference = float(np.mean(np.abs(crop(canvas, SCREENS[name][0]) - template)))
        if difference > LIMITS.get(name, MAX_DIFFERENCE):
            continue
        if name == 'donation':
            return name
        if best is None or difference < best[0]:
            best = (difference, name)
    return best[1] if best else None


class Operator:
    def __init__(self):
        enable_dpi()
        _, self.target, self.view = open_view()          # the view saved in the setup holds in menus
        self.templates = load()

    def canvas(self):
        if not bring_to_front(self.target):
            raise DesktopPause('Cannot bring the game window to the front.')
        time.sleep(0.2)
        return self.view.read().pixels

    def screen(self):
        return identify(self.canvas(), self.templates)

    def click(self, point, rest=0.25):
        """Hover over a canvas point, then click it (the game window is brought to the front)."""
        if not bring_to_front(self.target):
            raise DesktopPause('Cannot bring the game window to the front.')
        self.view.read()
        g, c = self.view.geometry, self.view.canvas
        x, y = g.x + c.x + point[0], g.y + c.y + point[1]
        self.target.check_point(x, y, foreground=True)
        desktop = di.virtual_desktop()
        di.send(di.mouse_events((x, y), desktop))
        time.sleep(rest)
        di.send(di.mouse_events((x, y), desktop, 'left', move=False))

    def start(self, timeout=20.0):
        """From the score screen or the setup screen to a starting match."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            screen = self.screen()
            if screen in ('score', 'main', 'donation'):
                self.click(SCREENS[screen][1])
            elif screen == 'setup':
                self.click(SCREENS['setup'][1])
                return 'started'
            time.sleep(1.0)
        return f'no known screen within {timeout:.0f} s'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['screen', 'click', 'start'])
    parser.add_argument('point', nargs='*', type=int)
    args = parser.parse_args()
    operator = Operator()
    if args.command == 'screen':
        print(operator.screen())
    elif args.command == 'click':
        if len(args.point) != 2:
            sys.exit('click needs X Y (canvas pixels)')
        operator.click(tuple(args.point))
        print('clicked', args.point)
    else:
        print(operator.start())


if __name__ == '__main__':
    main()
