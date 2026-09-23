"""Observe -> decide -> revalidate -> act. No browser-specific game logic here."""

from dataclasses import asdict
import json
import time

from .decision import choose_move


def fingerprint(board):
    return (board.rows, board.cols, board.status, board.mines_total,
            tuple(sorted((cell.row, cell.col, cell.state, cell.box.x, cell.box.y,
                          cell.box.width, cell.box.height) for cell in board.cells)))


class StaleObservation(RuntimeError):
    pass


class MinesweeperSession:
    def __init__(self, driver, brain=None, log=None, can_act=None):
        self.driver = driver
        self.brain = brain
        self.log = log
        self.steps = 0
        self.can_act = can_act

    def step(self):
        started = time.perf_counter()
        board = self.driver.observe()
        observed = time.perf_counter()
        if board.status != "playing":
            return {"status": board.status, "steps": self.steps}
        action, decision = choose_move(board, self.brain)
        decided = time.perf_counter()
        if action is None:
            return {"status": "stalled", "steps": self.steps}
        if self.can_act is not None and not self.can_act():
            return {"status": "paused", "steps": self.steps}
        current = self.driver.observe()
        if fingerprint(current) != fingerprint(board):
            raise StaleObservation("The board changed during the decision; no click was sent.")
        cell = next(cell for cell in current.cells if (cell.row, cell.col) == (action.row, action.col))
        if cell.state != "hidden":
            raise StaleObservation("The selected cell is no longer covered.")
        revalidated = time.perf_counter()
        # The driver also hit-tests the center immediately before mouse input.
        self.last_action = asdict(action)
        self.driver.click(cell, action.button)
        clicked = time.perf_counter()
        after = self.driver.wait_for_change(current)
        verified = time.perf_counter()
        self.steps += 1
        event = {"time": time.time(), "step": self.steps, "status": after.status,
                 "action": asdict(action), "box": asdict(cell.box), "decision": decision,
                 "timing_ms": {"observe": round((observed - started) * 1000, 3),
                               "decide": round((decided - observed) * 1000, 3),
                               "revalidate": round((revalidated - decided) * 1000, 3),
                               "input": round((clicked - revalidated) * 1000, 3),
                               "verify": round((verified - clicked) * 1000, 3),
                               "total": round((verified - started) * 1000, 3)},
                 "visible_before": [[c.row, c.col, c.state] for c in board.cells],
                 "visible_after": [[c.row, c.col, c.state] for c in after.cells]}
        if self.log:
            self.log.write(json.dumps(event) + "\n")
            self.log.flush()
        return event
