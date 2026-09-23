"""Bounded action selection through the existing SystemX HTTP API."""

from model_config import MODEL_NAME

import os
import time

import httpx

from .planner import candidates


class SystemOneDecision:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=30, client=None):
        headers = {}
        if os.environ.get("SYSTEMX_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["SYSTEMX_API_KEY"]
        self.client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)

    def close(self):
        self.client.close()

    def wait_ready(self, timeout=300, progress=None):
        started = time.monotonic()
        deadline = started + timeout
        last_message = None
        last_report = float("-inf")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"System One did not become ready within {timeout:g}s. Last state: {last_message}. Check the server console.")
            ready = False
            try:
                # /health reports loading normally, without filling server logs with 503s.
                response = self.client.get("/health", timeout=min(3, remaining))
                response.raise_for_status()
                status = response.json()
                if status.get("model") != MODEL_NAME:
                    raise RuntimeError("The configured endpoint is not the expected System One server.")
                if status.get("status") == "failed":
                    raise RuntimeError("System One initialization failed. Inspect the server console for the cause.")
                ready = status.get("ready") is True and status.get("status") == "ready"
                if not ready and status.get("status") != "loading":
                    raise RuntimeError("The server returned an unrecognized readiness state.")
                message = "System One ready; opening Minesweeper" if ready else "Loading AI model and warming up"
            except httpx.TransportError:
                message = f"Waiting for server at {self.client.base_url}; launch start.bat"
            now = time.monotonic()
            if progress and (message != last_message or now - last_report >= 5 or ready):
                progress(f"{message} [{now - started:.0f}s]." + (" Ctrl+C cancels." if not ready else ""))
                last_report = now
            last_message = message
            if ready:
                return
            time.sleep(min(2, max(0, deadline - time.monotonic())))

    def choose(self, board, options):
        visible = {(cell.row, cell.col): cell.state for cell in board.cells}
        symbols = {"hidden": "?", "flag": "F", "mine": "*"}
        state = {
            "game": "Minesweeper",
            "rows": board.rows,
            "columns": board.cols,
            "visible_board": ["".join(symbols.get(visible[r, c], visible[r, c])
                                       for c in range(board.cols)) for r in range(board.rows)],
            "legend": "? covered, F flagged, digits revealed adjacent mine counts. Coordinates start at 0.",
            "total_mines": board.mines_total,
        }
        criteria = {
            option.id: f"{'Flag' if option.button == 'right' else 'Reveal'} row {option.row}, column {option.col}. {option.reason}"
            for option in options
        }
        request = {
            "model": MODEL_NAME, "state": state,
            "questions": {"move": {
                "type": "choice",
                "instructions": "Choose a move to solve Minesweeper. Prefer proved safe actions. Among safe reveals prefer exposing useful new clues. If all are guesses, hidden mines cannot be known: choose a promising exploration, not an imagined certainty.",
                "criteria": criteria,
            }},
        }
        # Return control to the observation loop on busy/error, never reuse stale actions.
        response = self.client.post("/v1/systemone", json=request)
        response.raise_for_status()
        result = response.json()
        key = result.get("answers", {}).get("move", {}).get("choice")
        selected = next((option for option in options if option.id == key), None)
        if selected is None:
            raise ValueError("System One returned an action outside the current candidate list.")
        return selected, {"source": "systemone", "latency_ms": result.get("latency_ms"),
                          "answer": result["answers"]["move"]}


def choose_move(board, brain=None):
    options = candidates(board)
    if not options:
        return None, {"source": "terminal", "candidates": 0}
    if len(options) == 1 or brain is None:
        source = "only-candidate" if len(options) == 1 else "logic"
        return options[0], {"source": source, "candidates": len(options)}
    selected, details = brain.choose(board, options)
    return selected, {**details, "candidates": len(options)}
