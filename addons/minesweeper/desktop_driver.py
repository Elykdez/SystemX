"""Pixel-only adapter for the existing Minesweeper session."""

import time
import numpy as np
from desktop.target import DesktopPause
from .board import Box
from .visual.recognition import recognize
from .session import fingerprint
from .planner import candidates


# Require two matching frames at least one typical display refresh apart.
STABLE_FRAME_INTERVAL = .02
STABLE_FRAME_MAX_AGE = .1


class DesktopDriver:
    def __init__(self, capture, mouse, grid, profile, gate, generation, geometry, notify=None):
        self.capture, self.mouse = capture, mouse
        self.grid, self.profile = grid, profile
        self.gate, self.generation = gate, generation
        self.geometry = geometry
        self.profile_revision = profile.revision
        self.notify = notify
        self.last_frame = None
        self.last_board = None
        self.pending = None
        self._previous_observation = None
        self._board_pixels = None
        self._recognition = None
        self._recognition_settings = None
        self._boxes = tuple(grid.boxes())

    def parking_point(self, cell=None):
        g = self.grid
        if cell is not None:
            x, y = cell.box.center
            options = []
            if g.y >= 64:
                options.append((x, g.y - 64))
            if g.x >= 64:
                options.append((g.x - 64, y))
            if g.y + g.height + 64 < self.geometry.height:
                options.append((x, g.y + g.height + 64))
            if g.x + g.width + 64 < self.geometry.width:
                options.append((g.x + g.width + 64, y))
            if options:
                px, py = min(options, key=lambda p: (p[0] - x)**2 + (p[1] - y)**2)
                return self.geometry.x + px, self.geometry.y + py
        if g.y >= 24:
            return self.geometry.x + g.x + g.width / 2, self.geometry.y + g.y - min(64, g.y - 4)
        if g.x >= 24:
            return self.geometry.x + g.x - min(64, g.x - 4), self.geometry.y + g.y + g.height / 2
        raise DesktopPause('Leave at least 24 pixels above or left of the board for parking the pointer.')

    def prepare(self):
        self.mouse.move(*self.parking_point(), self.geometry, self.generation)
        time.sleep(.35)

    def _read(self):
        g = self.grid
        frame = self.capture.read((g.x, g.y, g.width, g.height))
        self.last_frame = frame
        if frame.geometry != self.geometry or self.profile.revision != self.profile_revision:
            raise DesktopPause('Geometry or profile changed. Capture and recalibrate before resuming.')
        for _, _, box in self._boxes:
            self.capture.target.check_point(frame.geometry.x + box.center[0], frame.geometry.y + box.center[1])
        pixels = g.crop(frame.pixels, Box(g.x, g.y, g.width, g.height), frame.origin)
        settings = (self.profile.threshold, self.profile.margin)
        changed = (self._board_pixels is None or settings != self._recognition_settings
                   or not np.array_equal(pixels, self._board_pixels))
        if changed:
            self._recognition = recognize(frame.pixels, g, self.profile, frame.origin)
            self._board_pixels = pixels.copy()
            self._recognition_settings = settings
        board, details = self._recognition
        if self.notify and changed:
            self.notify(frame, board, details)
        if board is None:
            unknown = sum(d['label'] is None for d in details)
            raise DesktopPause(f'{unknown} cells are uncertain. Inspect the preview and teach missing labels.')
        return board

    def observe(self):
        previous = self._previous_observation
        # A failed observation must not leave reusable stability evidence behind.
        self._previous_observation = None
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if previous is not None:
                age = time.monotonic() - previous[1]
                if age > STABLE_FRAME_MAX_AGE:
                    previous = None
                elif age < STABLE_FRAME_INTERVAL:
                    time.sleep(STABLE_FRAME_INTERVAL - age)
            board = self._read()
            observed_at = time.monotonic()
            current = fingerprint(board)
            if (previous is not None and observed_at - previous[1] <= STABLE_FRAME_MAX_AGE
                    and current == previous[0]):
                if board.status == 'playing':
                    try:
                        candidates(board)
                    except ValueError as error:
                        # A flag's background can paint before its glyph. Post-click
                        # verification may retry; pre-click validation still blocks input.
                        raise DesktopPause(f'Observed clues are inconsistent: {error}') from error
                self.last_board = board
                self._previous_observation = (current, observed_at)
                return board
            previous = (current, observed_at)
        raise DesktopPause('Board did not settle. Wait for animations, then resume.')

    def click(self, cell, button):
        if self.last_board is None or self.last_frame is None:
            raise DesktopPause('No validated observation before input.')
        expected = self.last_board
        fresh = self.observe()
        if fingerprint(fresh) != fingerprint(expected):
            raise DesktopPause('Board changed immediately before input.')
        if not self.gate.allowed(self.generation):
            raise DesktopPause('Pending decision was cancelled.')
        x, y = cell.box.center
        # Software cursor highlights can be captured even when hardware cursors are excluded.
        self.mouse.click(x + self.geometry.x, y + self.geometry.y, button, self.geometry, self.generation, park=self.parking_point(cell))
        self._previous_observation = None
        self.pending = (cell.row, cell.col, button)

    def wait_for_change(self, previous):
        if self.pending is None:
            raise DesktopPause('No click awaiting verification.')
        row, col, button = self.pending
        self.pending = None
        deadline = time.monotonic() + 3
        last_error = None
        while time.monotonic() < deadline:
            if not self.gate.allowed(self.generation):
                raise DesktopPause('Paused after input; click outcome will be reobserved before resuming.')
            try:
                current = self.observe()
            except DesktopPause as error:
                last_error = error
                time.sleep(STABLE_FRAME_INTERVAL)
                continue
            cell = next(c for c in current.cells if (c.row, c.col) == (row, col))
            if current.status == 'lost' or (button == 'left' and cell.state.isdigit()) or (button == 'right' and cell.state == 'flag'):
                return current
            time.sleep(STABLE_FRAME_INTERVAL)
        raise DesktopPause(f'Click result unconfirmed; no repeat sent. {last_error or "Selected cell did not change as expected."}')
