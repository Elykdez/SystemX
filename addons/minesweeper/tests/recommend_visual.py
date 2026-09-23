"""Read one saved screenshot and request one decision. Never sends mouse input."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
from addons.minesweeper.visual.recognition import Grid, Profile, recognize
from addons.minesweeper.decision import VonDecision, choose_move


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, nargs=4, required=True)
    parser.add_argument('--rows', type=int, default=9)
    parser.add_argument('--cols', type=int, default=9)
    parser.add_argument('--mines', type=int, default=10)
    parser.add_argument('--server', default='http://127.0.0.1:8002')
    parser.add_argument('--profile', type=Path, default=Path(__file__).parents[1] / 'visual/profiles/classic-web.json')
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error('Image could not be read')
    board, details = recognize(image, Grid(*args.grid, args.rows, args.cols, args.mines), Profile.load(args.profile))
    if board is None:
        result = dict(status='unknown', unknown=[d for d in details if d['label'] is None])
    else:
        brain = VonDecision(args.server)
        try:
            action, decision = choose_move(board, brain)
        finally:
            brain.close()
        result = dict(status=board.status, action=asdict(action) if action else None,
                      decision=decision, visible=[[c.row, c.col, c.state] for c in board.cells])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'visible'}))


if __name__ == '__main__':
    main()
