"""Read-only session recorder: canvas frames plus pointer-lock evidence. Sends no input."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from desktop.capture import Capture
from desktop.target import DesktopPause, enable_dpi
from .window import CANVAS_SIZE, Rect, find_game_window, locate_canvas, pointer_state

ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = ROOT / '.cache' / 'chronodivide' / 'recordings'


class GameView:
    """Capture of the game view only; finds the view again after geometry changes.

    In a match the sidebar's HUD anchors give the view exactly (calibration.find_game), whatever
    its size, and `on_found(rect, geometry)` hears of it (the player saves it in the setup).
    Elsewhere (menus, loading): the view saved in the setup (found in a match, or marked by hand)
    while the window keeps its client size; for a game drawn in a page (a browser), the page
    around it, pinned by its size; the place found before in this run; for a game that fills its
    window, the window's client area. `found_by` says which."""

    def __init__(self, target, foreground=False, relocate_seconds=None, profile=None, size=None, marked=None,
                 on_found=None):
        self.target, self.foreground = target, foreground
        self.relocate_seconds = relocate_seconds
        self.capture = Capture(target)
        self.canvas = self.geometry = None
        self.located_at = 0
        self.profile = profile
        self.embedded = True if profile is None else profile.embedded
        self.size = tuple(size or (profile.size if profile is not None else CANVAS_SIZE))
        self.marked = marked
        self.known = None                # (Rect, client size) of the last view found for sure
        self.found_by = None
        self.on_found = on_found
        self.error = ''

    def locate(self):
        frame = self.capture.read(foreground=self.foreground)
        self.canvas, self.geometry = self.find(frame.pixels), frame.geometry
        c, g = self.canvas, self.geometry
        if min(c.x, c.y) < 0 or c.right > g.width or c.bottom > g.height:
            self.canvas = None
            raise DesktopPause('The game canvas is clipped. Enlarge the browser or choose a smaller game '
                               'resolution so the entire canvas, including the bottom bar, is visible.')
        self.located_at = time.monotonic()
        if self.found_by == 'anchors' and self.on_found is not None:
            self.on_found(self.canvas, frame.geometry)
        return self.canvas

    def find(self, pixels):
        client = (pixels.shape[1], pixels.shape[0])
        if self.profile is not None:
            from .calibration import find_game
            found = find_game(pixels, self.profile)
            if found is not None:
                self.size, self.known, self.found_by = found.size, (found.rect, client), 'anchors'
                return found.rect
        if self.marked is not None and tuple(self.marked[1]) == client:
            self.found_by = 'setup'
            return self.marked[0]
        if not self.embedded:
            self.found_by = 'window'
            return Rect(0, 0, *client)               # a game that fills its window
        sizes = [self.size]
        if self.profile is not None and self.profile.size != self.size:
            sizes.append(self.profile.size)       # a saved fullscreen size may no longer fit the page
        for size in sizes:
            try:
                rect = locate_canvas(pixels, size)
            except DesktopPause:
                continue
            self.size, self.known, self.found_by = size, (rect, client), 'page'
            return rect
        if self.known is not None and self.known[1] == client:
            self.found_by = 'known'
            return self.known[0]
        raise DesktopPause('Cannot find the game view on the page. Keep the entire canvas, including '
                           'the bottom bar, visible at 100% zoom (Ctrl+0). Enlarge the window if needed; '
                           'if the canvas already fits, mark it in Setup.')

    def stale(self):
        return (self.canvas is None or self.target.geometry() != self.geometry or
                (self.relocate_seconds is not None and time.monotonic() - self.located_at > self.relocate_seconds))

    def read(self, region=None):
        """Frame of the canvas (or a canvas-relative region); pixels are BGR."""
        try:
            frame = self._read(region)
        except DesktopPause as error:
            self.error = str(error)
            raise
        self.error = ''
        return frame

    def _read(self, region=None):
        if self.stale():
            self.locate()
        c = self.canvas
        x, y, w, h = region or (0, 0, c.width, c.height)
        for px, py in ((x + w // 2, y + h // 2), (x + 1, y + 1), (x + w - 2, y + h - 2)):
            self.target.check_point(self.geometry.x + c.x + px, self.geometry.y + c.y + py)
        return self.capture.read((c.x + x, c.y + y, w, h), foreground=self.foreground)

    def close(self):
        self.capture.close()


def record(seconds, fps, output, quiet=False):
    enable_dpi()
    target = find_game_window()
    view = GameView(target, relocate_seconds=5)
    output.mkdir(parents=True, exist_ok=True)
    index = (output / 'index.jsonl').open('a', encoding='utf-8')
    previous, saved, skipped = None, 0, 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            tick = time.monotonic()
            canvas = None
            try:
                try:
                    frame = view.read()
                    canvas = view.canvas.tuple()
                except DesktopPause as error:
                    if 'covered' in str(error) or 'visible and not minimized' in str(error):
                        raise
                    # No page margin (e.g. a full-viewport canvas): keep the whole client area.
                    view.canvas = None
                    frame = view.capture.read(foreground=False)
            except DesktopPause as error:
                skipped += 1
                index.write(json.dumps(dict(t=round(tick - started, 3), skipped=str(error))) + '\n')
                time.sleep(.5)
                continue
            pointer = pointer_state()
            entry = dict(t=round(tick - started, 3), wall=datetime.now().isoformat(timespec='milliseconds'),
                         canvas=canvas, window=[frame.geometry.x, frame.geometry.y, frame.geometry.width,
                                                frame.geometry.height],
                         pointer=pointer, game_foreground=pointer['foreground'] == target.hwnd)
            if previous is None or previous.shape != frame.pixels.shape or not np.array_equal(previous, frame.pixels):
                name = f'frame-{saved:06d}.png'
                cv2.imwrite(str(output / name), frame.pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                entry['file'] = name
                previous, saved = frame.pixels, saved + 1
            index.write(json.dumps(entry) + '\n')
            index.flush()
            if not quiet and saved % 20 == 0 and 'file' in entry:
                print(f'{entry["t"]:.0f}s: {saved} frames, canvas {canvas}', flush=True)
            time.sleep(max(0, 1 / fps - (time.monotonic() - tick)))
    finally:
        index.close()
        view.close()
    return dict(saved=saved, skipped=skipped, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--fps', type=float, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or RECORDINGS / datetime.now().strftime('%Y%m%d-%H%M%S')
    print(json.dumps(record(args.seconds, args.fps, output)))


if __name__ == '__main__':
    main()
