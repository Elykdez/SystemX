"""Unit boxes on recorded frames: what the agent recognizes, drawn for review, or exported as
labels for training a detector.

    python -m addons.chronodivide.tools.units FOLDER [--out DIR] [--every N] [--export DIR]

FOLDER is a run's frames folder (.cache/chronodivide/runs/<time>/frames). Each frame with units
in view is written to --out (default: FOLDER/../units-review) with the boxes and labels the
inspector shows: our selected units found by their health bars and named from the units built
in that run (its summary.json), the enemy's objects by house color (perception/unitboxes.py).

--export DIR writes a YOLO-format dataset instead: images/, labels/ ('class cx cy w h' per box,
normalized) and classes.txt. The labels come from the rules above, so they cover our selected
units only: review them in a labeling tool (unselected units, unit types) before training a
small detector (for example YOLOv8n, a few ms per frame on an RTX 3070) that names every unit.
"""

import argparse
from collections import Counter
import glob
import json
from pathlib import Path

import cv2

from ..perception.unitboxes import draw, enemy_objects, our_units, recognize
from ..world import load_knowledge

CLASSES = ('our_infantry', 'our_vehicle', 'enemy_unit', 'enemy_building')


def built_in_run(folder):
    """Units built in the run (from its summary.json), for naming our units; empty when unknown."""
    summary = Path(folder).parent / 'summary.json'
    try:
        units = json.loads(summary.read_text(encoding='utf-8'))['world']['units']
    except (OSError, KeyError, ValueError):
        return {}
    return {item: n for item, n in units.items() if item not in ('war_miner', 'engineer', 'mcv')}


def class_of(unit):
    if unit.side == 'ours':
        return CLASSES.index('our_vehicle' if unit.kind == 'vehicle' else 'our_infantry')
    return CLASSES.index('enemy_building' if unit.kind == 'building' else 'enemy_unit')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder')
    parser.add_argument('--out', help='Folder for the annotated frames (default: FOLDER/../units-review).')
    parser.add_argument('--every', type=int, default=1, help='Use every Nth frame.')
    parser.add_argument('--export', help='Write a YOLO-format dataset here instead of annotated frames.')
    args = parser.parse_args()
    items = load_knowledge()['items']
    built = built_in_run(args.folder)
    out = Path(args.export or args.out or Path(args.folder).parent / 'units-review')
    if args.export:
        (out / 'images').mkdir(parents=True, exist_ok=True)
        (out / 'labels').mkdir(parents=True, exist_ok=True)
        (out / 'classes.txt').write_text('\n'.join(CLASSES) + '\n', encoding='utf-8')
    else:
        out.mkdir(parents=True, exist_ok=True)
    frames, written, totals = 0, 0, Counter()
    for number, path in enumerate(sorted(glob.glob(str(Path(args.folder) / 'frame-*.png')))):
        if number % args.every:
            continue
        canvas = cv2.imread(path)
        if canvas is None or canvas.shape[:2] != (768, 1024):
            continue
        frames += 1
        ours = recognize(our_units(canvas), built, items)
        units = ours + enemy_objects(canvas)
        if not ours:
            continue                       # nothing selected: nothing of ours to show or learn from
        written += 1
        totals.update(u.name or u.kind for u in ours)
        name = Path(path).stem
        if args.export:
            cv2.imwrite(str(out / 'images' / f'{name}.png'), canvas)
            height, width = canvas.shape[:2]
            lines = [f'{class_of(u)} {(u.box[0] + u.box[2]) / 2 / width:.5f} {(u.box[1] + u.box[3]) / 2 / height:.5f} '
                     f'{(u.box[2] - u.box[0]) / width:.5f} {(u.box[3] - u.box[1]) / height:.5f}' for u in units]
            (out / 'labels' / f'{name}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        else:
            cv2.imwrite(str(out / f'{name}.png'), draw(canvas.copy(), units, items))
    print(f'{written} of {frames} frames had selected units -> {out}')
    print('our units seen (frames x units):', dict(totals))


if __name__ == '__main__':
    main()
