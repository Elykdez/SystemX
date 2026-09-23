"""Minesweeper's rendered DOM is the first perception adapter."""

import time

from .observation import OBSERVE_SCRIPT, parse_observation
from .session import fingerprint, StaleObservation


class MinesweeperBrowser:
    def __init__(self, page, action_timeout=3):
        self.page = page
        self.action_timeout = action_timeout

    def observe(self):
        return parse_observation(self.page.evaluate(OBSERVE_SCRIPT))

    def click(self, cell, button):
        if button not in ("left", "right"):
            raise ValueError("Unsupported mouse button")
        label = f"Row {cell.row + 1}, column {cell.col + 1}, closed"
        # Hit-test immediately before input; reject overlays or geometry changes.
        expected = {"label": label, "x": cell.box.x, "y": cell.box.y,
                    "width": cell.box.width, "height": cell.box.height}
        allowed = self.page.evaluate("""expected => {
            const x = expected.x + expected.width / 2;
            const y = expected.y + expected.height / 2;
            const hit = document.elementFromPoint(x, y);
            const button = hit && hit.closest('#board button');
            if (!button || button.disabled || button.getAttribute('aria-label') !== expected.label) return false;
            const rect = button.getBoundingClientRect();
            return ['x','y','width','height'].every(key => Math.abs(rect[key] - expected[key]) < 0.5);
        }""", expected)
        if not allowed:
            raise StaleObservation("Cell is obscured, offscreen, disabled, or moved; no click sent.")
        self.page.mouse.click(*cell.box.center, button=button)

    def wait_for_change(self, previous):
        deadline = time.monotonic() + self.action_timeout
        before = fingerprint(previous)
        while time.monotonic() < deadline:
            current = self.observe()
            if fingerprint(current) != before:
                if (current.rows, current.cols) != (previous.rows, previous.cols):
                    raise StaleObservation("Board dimensions changed after input.")
                old_states = {(cell.row, cell.col): cell.state for cell in previous.cells}
                new_states = {(cell.row, cell.col): cell.state for cell in current.cells}
                if new_states == old_states and current.status == previous.status:
                    raise StaleObservation("Only board geometry changed; the click result is unconfirmed.")
                return current
            self.page.wait_for_timeout(50)
        raise TimeoutError("No visible change after clicking; stopped instead of repeating input.")
