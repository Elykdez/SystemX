"""Extract and cluster sidebar cameo appearances from recordings for labeling.

Writes a numbered contact sheet plus clusters.npz (one representative crop per
cluster). Labels are then assigned in cameo-labels.json and turned into templates.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..layout import Layout, PROFILES
from .sheet import canvas_frames


def slot_crops(image, layout):
    for index, box in enumerate(layout.grid.boxes()):
        yield index, box.crop(image)


def cluster(folders, layout, threshold=0.012, limit=400):
    empty = None
    reps, counts, first = [], [], []
    for folder in folders:
        for number, t, image in canvas_frames(folder):
            if image.shape[:2] != (layout.canvas[1], layout.canvas[0]):
                continue
            for index, crop in slot_crops(image, layout):
                crop = crop.astype(np.int16)
                if empty is not None and np.mean(np.abs(crop - empty[index])) / 255 < 0.02:
                    continue
                for c, rep in enumerate(reps):
                    if np.mean(np.abs(crop - rep)) / 255 < threshold:
                        counts[c] += 1
                        break
                else:
                    if len(reps) < limit:
                        reps.append(crop)
                        counts.append(1)
                        first.append((str(folder), number, index))
            if empty is None and number >= 0:
                # First frame with a sidebar: all-empty slots define the empty appearance.
                crops = [c.astype(np.int16) for _, c in slot_crops(image, layout)]
                if np.std([np.mean(c) for c in crops]) < 3:
                    empty = crops
                    reps.clear(), counts.clear(), first.clear()
    return reps, counts, first


def contact(reps, counts, output, columns=12, scale=2):
    tiles = []
    for number, (rep, count) in enumerate(zip(reps, counts)):
        tile = cv2.resize(rep.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        tile = np.pad(tile, ((14, 2), (2, 2), (0, 0)))
        cv2.putText(tile, f'{number}:{count}', (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 255), 1)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + columns] + [np.zeros_like(tiles[0])] * (columns - len(tiles[i:i + columns])))
            for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(output), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layout', default=str(PROFILES / 'layout-1024x768.json'))
    parser.add_argument('--threshold', type=float, default=0.012)
    args = parser.parse_args()
    layout = Layout.load(args.layout)
    reps, counts, first = cluster(args.folders, layout, args.threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'clusters.npz', *[r.astype(np.uint8) for r in reps])
    (args.out / 'clusters.json').write_text(json.dumps([dict(id=i, count=c, first=f) for i, (c, f) in
                                                         enumerate(zip(counts, first))], indent=1), encoding='utf-8')
    contact(reps, counts, args.out / 'clusters.png')
    print(len(reps), 'clusters')


if __name__ == '__main__':
    main()
