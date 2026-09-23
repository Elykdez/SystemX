"""The unit detector: build its dataset from recorded games, train it, measure it, export it.

    python -m addons.chronodivide.tools.detector build     labeled views from runs, pairs and census pairs
    python -m addons.chronodivide.tools.detector train     train on all runs but the held-out ones
    python -m addons.chronodivide.tools.detector eval      recall and precision on the held-out views
    python -m addons.chronodivide.tools.detector export    TorchScript + metadata into profiles/
    python -m addons.chronodivide.tools.detector review    detections drawn on held-out views

Every image is a frame as the game drew it: nothing is painted out, recolored or mirrored. The
labels come from the game's own marks. A selected unit shows a health bar whose length gives its
size class (perception/unitboxes.py), so a frame with selected units labels them as they are,
bars and all. The same units unselected are labeled from a neighboring frame where they are
selected: a pair of frames with the camera where it was (phase correlation) and the unit's
sprite unchanged (it has not moved). Pairs are found in the runs' recorded frames (a second
apart) and recorded on purpose by the army's census (the view just before P selects the army at
home, the view with it selected, and the view just after it is deselected: PairRecorder).
Our house-color patches that no bar claims may be unselected units or parts of buildings, and
the enemy's are not labeled at all: both are left out of the loss (ignore boxes), neither
counted as units nor as background.

Runs are split, not frames: held-out runs (--holdout) never train. Everything is written under
.cache/chronodivide/detector/ except the exported model (profiles/detector-units.pt and .json).
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import threading
import time

import cv2
import numpy as np

from ..perception.modes import ModeReader
from ..perception.scene import CameraShift
from ..perception.unitboxes import our_units
from ..perception.units import enemy_blobs, house_blobs

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.cache' / 'chronodivide' / 'detector'
DATASET = WORK / 'dataset'
CENSUS = WORK / 'census'
RUNS = ROOT / '.cache' / 'chronodivide' / 'runs'
FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures'
PROFILES = Path(__file__).resolve().parents[1] / 'profiles'
MODEL = PROFILES / 'detector-units.pt'
TACTICAL = (0, 0, 850, 736)
CLASSES = ('infantry', 'vehicle')
UNIT_PATCH = 400                 # our color patches smaller than this may be unselected units: ignored
STILL = 12.0                     # mean channel difference below which a unit's sprite has not changed


# -- labels from the game's own marks ----------------------------------------------------
def _inside(point, box, margin=4):
    return box[0] - margin <= point[0] <= box[2] + margin and box[1] - margin <= point[1] <= box[3] + margin


def ignore_boxes(canvas, boxes, region=TACTICAL):
    """Where labels may be missing: our small patches that no labeled box holds, and every enemy patch."""
    ignore = []
    for area, center, (x, y, w, h) in house_blobs(canvas, region):
        if area < UNIT_PATCH and max(w, h) < 45 and not any(_inside(center, b) for b in boxes):
            ignore.append([x - 12, y - 16, x + w + 12, y + h + 12])
    for area, center, (x, y, w, h) in enemy_blobs(canvas, region):
        ignore.append([x - 12, y - 16, x + w + 12, y + h + 12])
    return ignore


def label(canvas, region=TACTICAL):
    """([[x0, y0, x1, y1, class]], ignore boxes) of a frame with selected units, or None."""
    units = our_units(canvas, region)
    if not units:
        return None
    boxes = [[*u.box, CLASSES.index(u.kind)] for u in units]
    return boxes, ignore_boxes(canvas, [b[:4] for b in boxes], region)


def still(selected, other, box, threshold=STILL):
    """Has the unit in `box` stayed put between the two frames? Its sprite (the lower part of the
    box: the bar and team badge are drawn above it) must look the same in both."""
    x0, y0, x1, y1 = box
    top = y0 + (y1 - y0) * 3 // 10
    a = selected[top:y1, x0:x1].astype(np.int16)
    b = other[top:y1, x0:x1].astype(np.int16)
    return a.size > 0 and float(np.mean(np.abs(a - b))) <= threshold


def pair_labels(selected, other, region=TACTICAL):
    """Labels for `other`, a frame of the same view without the selection, from `selected`: the units
    that stayed put, with ignore boxes for the ones that moved. None unless the camera is where it
    was and `other` shows no selection."""
    x, y, w, h = region
    if our_units(other, region):
        return None                                  # not an unselected view
    camera = CameraShift()
    camera(selected, region)
    if camera(other, region) != (0, 0):
        return None                                  # the camera moved or jumped
    units = our_units(selected, region)
    kept = [[*u.box, CLASSES.index(u.kind)] for u in units if still(selected, other, u.box)]
    if not kept:
        return None
    moved = [[u.box[0] - 12, u.box[1] - 12, u.box[2] + 12, u.box[3] + 12] for u in units
             if not still(selected, other, u.box)]
    return kept, ignore_boxes(other, [b[:4] for b in kept], region) + moved


class PairRecorder:
    """Saves the census's frames (army.py) as labeled views of unselected units: called on the
    control loop with three frames, it labels and writes them on a thread of its own."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.saved = 0

    def __call__(self, before, selected, after):
        threading.Thread(target=self._write, args=(before, selected, after), daemon=True).start()

    def _write(self, before, selected, after):
        stamp = int(time.time() * 1000)
        for name, frame in (('before', before), ('after', after)):
            if frame is None or selected is None:
                continue
            labels = pair_labels(selected, frame)
            if labels is None:
                continue
            self.folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.folder / f'{stamp}-{name}.png'), frame[:TACTICAL[3], :TACTICAL[2]],
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            (self.folder / f'{stamp}-{name}.json').write_text(json.dumps(dict(boxes=labels[0], ignore=labels[1])))
            self.saved += 1


def _write(folder, name, run, kind, image, labels):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f'{name}.png'), image[:TACTICAL[3], :TACTICAL[2]], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    (folder / f'{name}.json').write_text(json.dumps(dict(run=run, kind=kind, boxes=labels[0], ignore=labels[1])))


def build(args):
    """Labeled views from the runs' frames (selected, and unselected paired with a neighbor a second
    away), the test fixtures (selected), and the census pairs recorded in live games."""
    modes = ModeReader.load(PROFILES / 'hud-1024x768.npz')
    counts = dict(selected=0, pair=0, census=0)
    sources = [(run.name, sorted((run / 'frames').glob('*.png'))) for run in sorted(RUNS.iterdir())
               if (run / 'frames').exists()]
    sources.append(('fixtures', sorted(FIXTURES.glob('*.png'))))
    for run, frames in sources:
        folder = DATASET / run
        views = []
        for path in frames:
            canvas = cv2.imread(str(path))
            ok = canvas is not None and canvas.shape[:2] == (768, 1024) and modes.read(canvas)['mode'] == 'hud'
            views.append((path, canvas if ok else None))
        for index, (path, canvas) in enumerate(views):
            if canvas is None:
                continue
            labels = label(canvas)
            if labels is None:
                continue
            _write(folder, path.stem, run, 'selected', canvas, labels)
            counts['selected'] += 1
            if run == 'fixtures':
                continue                              # fixtures are single frames, not a sequence
            for step in (-1, 1):
                if not 0 <= index + step < len(views) or views[index + step][1] is None:
                    continue
                other_path, other = views[index + step]
                paired = pair_labels(canvas, other)
                if paired is not None and not (folder / f'{other_path.stem}.json').exists():
                    _write(folder, other_path.stem, run, 'pair', other, paired)
                    counts['pair'] += 1
    for meta in sorted(CENSUS.glob('*/*.json')):
        image = cv2.imread(str(meta.with_suffix('.png')))
        if image is None:
            continue
        data = json.loads(meta.read_text())
        _write(DATASET / f'census-{meta.parent.name}', meta.stem, f'census-{meta.parent.name}', 'census', image,
               (data['boxes'], data['ignore']))
        counts['census'] += 1
    print(f'labeled views in {DATASET}: {counts}')


def load(holdout, dataset=DATASET):
    train, test = [], []
    for meta in sorted(dataset.glob('*/*.json')):
        data = json.loads(meta.read_text())
        image = cv2.imread(str(meta.with_suffix('.png')))
        if image is None:
            continue
        sample = dict(image=image, boxes=np.array(data['boxes'], np.float32).reshape(-1, 5),
                      ignore=np.array(data['ignore'], np.float32).reshape(-1, 4), run=data['run'],
                      kind=data.get('kind', 'selected'), name=meta.stem)
        (test if any(h in data['run'] for h in holdout) else train).append(sample)
    return train, test


# -- training ----------------------------------------------------------------------------
def targets(boxes, ignore, size, stride=4, classes=len(CLASSES)):
    """Heatmap (Gaussian peaks at unit centers), sizes and offsets at the centers, and the ignore map."""
    h, w = size[0] // stride, size[1] // stride
    heat = np.zeros((classes, h, w), np.float32)
    wh = np.zeros((2, h, w), np.float32)
    off = np.zeros((2, h, w), np.float32)
    mask = np.zeros((h, w), np.float32)
    skip = np.zeros((h, w), np.float32)
    for x0, y0, x1, y1 in ignore:
        skip[max(0, int(y0 // stride)):max(0, int(np.ceil(y1 / stride))),
             max(0, int(x0 // stride)):max(0, int(np.ceil(x1 / stride)))] = 1
    ys, xs = np.mgrid[0:h, 0:w]
    for x0, y0, x1, y1, cls in boxes:
        cx, cy = (x0 + x1) / 2 / stride, (y0 + y1) / 2 / stride
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        sigma = max(0.8, 0.15 * np.sqrt((x1 - x0) * (y1 - y0)) / stride)
        heat[int(cls)] = np.maximum(heat[int(cls)], np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma ** 2)))
        heat[int(cls), iy, ix] = 1.0
        wh[:, iy, ix] = ((x1 - x0) / stride, (y1 - y0) / stride)
        off[:, iy, ix] = (cx - ix, cy - iy)
        mask[iy, ix] = 1
        skip[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = 0          # never ignore a labeled unit
    return heat, wh, off, mask, skip


def sample_crop(sample, crop, rng):
    """A crop of a view as it was drawn (most often around a labeled unit), with its labels."""
    image, boxes, ignore = sample['image'], sample['boxes'], sample['ignore']
    height, width = image.shape[:2]
    if len(boxes) and rng.random() < 0.8:
        x0, y0, x1, y1, _ = boxes[rng.randrange(len(boxes))]
        left = int((x0 + x1) / 2 - rng.uniform(0.2, 0.8) * crop)
        top = int((y0 + y1) / 2 - rng.uniform(0.2, 0.8) * crop)
    else:
        left, top = rng.randrange(0, max(1, width - crop)), rng.randrange(0, max(1, height - crop))
    left, top = min(max(0, left), width - crop), min(max(0, top), height - crop)
    shift = np.array([left, top, left, top], np.float32)
    kept = [b for b in boxes if left <= (b[0] + b[2]) / 2 < left + crop and top <= (b[1] + b[3]) / 2 < top + crop]
    boxes = np.array([[*(b[:4] - shift), b[4]] for b in kept], np.float32).reshape(-1, 5)
    ignore = (ignore - shift).reshape(-1, 4) if len(ignore) else ignore
    return np.ascontiguousarray(image[top:top + crop, left:left + crop]), boxes, ignore


def train(args):
    import torch
    from ..perception.detector_net import UnitNet, focal_loss, normalize
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    samples, held = load(args.holdout)
    if not samples:
        raise SystemExit('No training data: run "build" first.')
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    net = UnitNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps, pct_start=0.1)
    kinds = {k: sum(s['kind'] == k for s in samples) for k in ('selected', 'pair', 'census')}
    print(f'{len(samples)} training views {kinds}, {sum(len(s["boxes"]) for s in samples)} units; '
          f'{len(held)} held out; {sum(p.numel() for p in net.parameters())} parameters on {device}')
    started = time.time()
    for step in range(args.steps):
        crops = [sample_crop(samples[rng.randrange(len(samples))], args.crop, rng) for _ in range(args.batch)]
        parts = [targets(b, i, (args.crop, args.crop)) for _, b, i in crops]
        x = normalize(torch.from_numpy(np.stack([c[0] for c in crops])).to(device))
        heat_t, wh_t, off_t, mask_t, skip_t = (torch.from_numpy(np.stack([p[k] for p in parts])).to(device)
                                               for k in range(5))
        heat, wh, off = net(x)
        positive = mask_t.unsqueeze(1)
        count = positive.sum().clamp(min=1)
        loss = focal_loss(heat, heat_t, skip_t.unsqueeze(1)) + \
            0.1 * (torch.abs(wh - wh_t) * positive).sum() / count + (torch.abs(off - off_t) * positive).sum() / count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f'step {step:5d}  loss {loss.item():.3f}  {time.time() - started:.0f} s', flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), holdout=args.holdout, steps=args.steps, trained=datetime.now().isoformat(),
                    views=len(samples), kinds=kinds, units=int(sum(len(s['boxes']) for s in samples))),
               WORK / 'unitnet.pth')
    print('saved', WORK / 'unitnet.pth')


# -- measuring and exporting -------------------------------------------------------------------
def scripted(path=WORK / 'unitnet.pth', device='cpu'):
    import torch
    from ..perception.detector_net import Exported, UnitNet
    checkpoint = torch.load(path, map_location='cpu')
    net = UnitNet()
    net.load_state_dict(checkpoint['state'])
    return torch.jit.script(Exported(net.eval())).to(device).eval(), checkpoint


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(rows, boxes, ignore, threshold, min_iou=0.3):
    """(true positives, class right, false positives, labeled units) for one view; detections centered
    in an ignore box are not counted either way."""
    found = [r for r in rows if r[4] >= threshold]
    used, hits, right, false = set(), 0, 0, 0
    for row in sorted(found, key=lambda r: -r[4]):
        best = max(((iou(row, b), i) for i, b in enumerate(boxes) if i not in used), default=(0.0, None))
        if best[0] >= min_iou:
            used.add(best[1])
            hits += 1
            right += int(round(row[5])) == int(boxes[best[1]][4])
        elif not any(_inside(((row[0] + row[2]) / 2, (row[1] + row[3]) / 2), g, 0) for g in ignore):
            false += 1
    return hits, right, false, len(boxes)


def evaluate(args):
    """Held-out views, reported separately for selected units (bars shown) and unselected ones."""
    import torch
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model, checkpoint = scripted(device=device)
    _, held = load(checkpoint['holdout'])
    if not held:
        raise SystemExit('No held-out views.')
    outputs, times = [], []
    with torch.inference_mode():
        for sample in held:
            image = torch.from_numpy(sample['image']).to(device)
            started = time.perf_counter()
            rows = model(image)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
            outputs.append(rows.cpu().numpy())
    report = dict(holdout=checkpoint['holdout'], device=device, ms_median=round(float(np.median(times[1:] or times)), 1))
    for group, kinds in (('selected', ('selected',)), ('unselected', ('pair', 'census'))):
        chosen = [(rows, s) for rows, s in zip(outputs, held) if s['kind'] in kinds]
        if not chosen:
            continue
        part = dict(views=len(chosen), units=int(sum(len(s['boxes']) for _, s in chosen)))
        for threshold in (0.3, 0.4, 0.5):
            totals = np.sum([match(rows, s['boxes'], s['ignore'], threshold) for rows, s in chosen], axis=0)
            hits, right, false, labeled = (int(v) for v in totals)
            part[f'at_{threshold}'] = dict(recall=round(hits / max(1, labeled), 3),
                                           precision=round(hits / max(1, hits + false), 3),
                                           class_right=round(right / max(1, hits), 3), false=false)
        report[group] = part
    print(json.dumps(report, indent=2))
    (WORK / 'eval.json').write_text(json.dumps(report, indent=2))
    return report


def export(args):
    import torch
    model, checkpoint = scripted()
    report = json.loads((WORK / 'eval.json').read_text()) if (WORK / 'eval.json').exists() else None
    torch.jit.save(model, str(MODEL))
    meta = dict(name='ra2-units', classes=list(CLASSES), input='uint8 BGR image (H, W, 3), the tactical view',
                output='rows [x0, y0, x1, y1, score, class], best first', threshold=args.threshold,
                sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(), trained=checkpoint['trained'],
                steps=checkpoint['steps'], views=checkpoint['views'], kinds=checkpoint.get('kinds'),
                units=checkpoint['units'], holdout=checkpoint['holdout'], evaluation=report)
    MODEL.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'exported {MODEL} ({MODEL.stat().st_size // 1024} KiB)')


def review(args):
    import torch
    model, checkpoint = scripted()
    _, held = load(checkpoint['holdout'])
    out = WORK / 'review'
    out.mkdir(parents=True, exist_ok=True)
    for sample in held[::max(1, len(held) // 12)]:
        image = sample['image'].copy()                   # a copy for drawing on; the view is untouched
        with torch.inference_mode():
            rows = model(torch.from_numpy(sample['image'])).numpy()
        for x0, y0, x1, y1, cls in sample['boxes']:
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)
        for x0, y0, x1, y1, score, cls in rows[rows[:, 4] >= args.threshold]:
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255) if cls else (255, 128, 0), 2)
            cv2.putText(image, f'{CLASSES[int(cls)][:3]} {score:.2f}', (int(x0), int(y0) - 2), cv2.FONT_HERSHEY_PLAIN,
                        0.8, (255, 255, 255), 1)
        cv2.imwrite(str(out / f'{sample["kind"]}-{sample["run"]}-{sample["name"]}.jpg'), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('review frames in', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'train', 'eval', 'export', 'review'])
    parser.add_argument('--holdout', nargs='+', default=['fixtures'], help='Runs kept out of training.')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--crop', type=int, default=320)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    dict(build=build, train=train, eval=evaluate, export=export, review=review)[args.command](args)


if __name__ == '__main__':
    main()
