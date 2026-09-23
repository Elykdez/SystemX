"""Contact sheets of recorded canvas frames (optionally one region) for quick review."""

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

from ..window import locate_canvas


def canvas_frames(folder):
    """Yield (index, time, canvas BGR) for every saved frame, cropping full-client captures."""
    index = {}
    for line in (Path(folder) / 'index.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if 'file' in row:
            index[row['file']] = row
    rect = None
    for path in sorted(glob.glob(str(Path(folder) / 'frame-*.png'))):
        image = cv2.imread(path)
        row = index.get(Path(path).name, {})
        if row.get('canvas') is None:
            try:
                rect = locate_canvas(image)
            except Exception:
                if rect is None:
                    continue
            if rect.height > 768 and rect.width == 1024:
                rect = type(rect)(rect.x, rect.bottom - 768, 1024, 768)
            image = image[rect.y:rect.bottom, rect.x:rect.right]
        yield int(Path(path).stem.split('-')[1]), row.get('t'), image


def sheet(folder, output, every=10, region=None, width=256, columns=6, start=0, stop=None):
    tiles = []
    for number, t, image in canvas_frames(folder):
        if number < start or (stop is not None and number > stop) or number % every:
            continue
        if region:
            x, y, w, h = region
            image = image[y:y + h, x:x + w]
        scale = width / image.shape[1]
        tile = cv2.resize(image, (width, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f'{number} t={t}', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        raise SystemExit('No frames matched.')
    height = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, height - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    rows = [np.hstack(tiles[i:i + columns] + [np.zeros_like(tiles[0])] * (columns - len(tiles[i:i + columns])))
            for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(output), np.vstack(rows))
    return len(tiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('output')
    parser.add_argument('--every', type=int, default=10)
    parser.add_argument('--region', type=int, nargs=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--columns', type=int, default=6)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()
    print(sheet(args.folder, args.output, args.every, args.region, args.width, args.columns, args.start, args.stop))


if __name__ == '__main__':
    main()
