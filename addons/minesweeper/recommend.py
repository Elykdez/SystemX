"""Recommend one move from a saved visible observation, without browser control."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .decision import VonDecision, choose_move
from .observation import parse_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--policy", choices=("von", "logic"), default="von")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    board = parse_observation(json.loads(args.observation.read_text(encoding="utf-8")))
    brain = VonDecision(args.server) if args.policy == "von" else None
    try:
        action, details = choose_move(board, brain)
        box = next((asdict(cell.box) for cell in board.cells
                    if action and (cell.row, cell.col) == (action.row, action.col)), None)
        print(json.dumps({"status": board.status, "action": asdict(action) if action else None,
                          "box": box, "decision": details}))
    finally:
        if brain:
            brain.close()


if __name__ == "__main__":
    main()
