"""Conservative, once-per-building repair from a freshly selected health bar.

Only calibrated building appearances in the side's repair list are eligible. Colored diagonal pips
report a health band, not an invented percentage. A repair latch never resets
automatically: clicking an already repairing structure would cancel its repair.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .executor import Step
from .perception.buildings import default_recognizer
from .tasks import Act, Look


@lru_cache(maxsize=1)
def allied_mode_icons():
    path = Path(__file__).with_name('profiles') / 'repair-icons-allied-1024x768.npz'
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].astype(np.int16) for key in data.files}


def mode_icon(canvas, layout, name):
    box = layout.regions.get(name)
    if box is None:
        return None
    x0, x1 = ((.50, .87) if name == 'repair' else (.23, .62))
    crop = canvas[box.y + round(box.h * .15):box.y + round(box.h * .95),
                  box.x + round(box.w * x0):box.x + round(box.w * x1)]
    b, g, r = cv2.split(crop.astype(np.int16))
    if np.count_nonzero((r > 150) & (g > 110) & (r-b > 55) & (g-b > 45)) >= 25:
        return 'active'
    if np.count_nonzero((r > 120) & (r-g > 35) & (r-b > 35)) >= 25:
        return 'inactive'
    # Allied buttons highlight cyan rather than Soviet yellow/red.
    for state in ('active', 'inactive'):
        template = allied_mode_icons().get(name+'_'+state)
        if template is not None and crop.shape == template.shape:
            if np.mean(np.max(abs(crop.astype(np.int16)-template), axis=2) <= 20) >= .8:
                return state
    return None


def selected_health(canvas, region, before=None, min_length=20, diagonal=None):
    """Find a newly exposed, long isometric health-pip line near the selection.

The live client renders building pips on slope -1/2; ordinary horizontal unit
bars and building colors are insufficient evidence. Unknown/short bars abstain.
    """
    x0, y0, x1, y1 = region
    crop = canvas[y0:y1, x0:x1].astype(np.int16)
    b, g, r = cv2.split(crop)
    masks = dict(green=(g > 150) & (g-r > 70) & (g-b > 60),
                 yellow=(r > 180) & (g > 160) & (b < 140) & (abs(r-g) < 70),
                 red=(r > 170) & (r-g > 85) & (r-b > 85))
    changed = (np.ones(b.shape, bool) if before is None else
               np.max(abs(crop-before[y0:y1, x0:x1].astype(np.int16)), axis=2) > 35)
    if diagonal is not None:
        ax, ay, bx, by = diagonal
        yy, xx = np.indices(b.shape)
        changed &= (xx+x0 >= ax-3) & (xx+x0 <= bx+3)
        changed &= abs(yy+y0-(ay+(xx+x0-ax)*(by-ay)/(bx-ax))) <= 4
    if before is None or diagonal is None:
        min_length = max(20, min_length)
    found = []
    for level, mask in masks.items():
        mask = (mask & changed).astype(np.uint8)
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, 8,
                                minLineLength=min_length-4, maxLineGap=3)
        for line in (() if lines is None else lines[:, 0]):
            ax, ay, bx, by = map(int, line)
            if bx-ax >= min_length and -.60 <= (by-ay)/(bx-ax) <= -.40:
                found.append((bx-ax, level))
    return max(found)[1] if found else None


class Repair:
    def __init__(self, ctx, runner, book, clock):
        self.ctx, self.runner, self.clock = ctx, runner, clock
        self.policy = book.data['playbook'].get('repair', {})
        self.next_scan = 0
        self.latched = []
        self.cleanup_pending = False
        self.cleanup_attempts = 0
        self.scan_index = 0
        self.scan_active = False

    def _inactive(self):
        return all(mode_icon(self.ctx.last_canvas, self.ctx.layout, name) == 'inactive'
                   for name in ('repair', 'sell'))

    def _cleanup(self):
        # Right-click is idempotent even when a repair click already exited K mode.
        self.cleanup_attempts += 1
        yield Act([Step.click(self.ctx.clear_point, 'right'), Step.move(self.ctx.park),
                   Step.wait(self.policy['settle_ms'])])
        self.cleanup_pending = not (yield Look(lambda *_: self._inactive()))
        if self.cleanup_pending and self.cleanup_attempts >= self.policy['cleanup_attempts']:
            self.ctx.world.note('repair-cleanup-blocked', reason='inactive mode icons not verified')

    def step(self, snapshot, act=True):
        """Return True while this controller owns input, including uncertain cleanup."""
        if not act or not self.policy.get('enabled') or tuple(self.ctx.layout.canvas) != (1024, 768):
            return False
        if any(t.name.startswith('repair:') for t in self.runner.tasks.values()):
            return True
        if self.cleanup_pending:
            if not self.runner.tasks and self.cleanup_attempts < self.policy['cleanup_attempts']:
                self.runner.start('repair:cleanup', 'sidebar', self._cleanup(), priority=0)
            return True
        now = self.clock()
        if self.scan_active:
            self.scan_active = False
            self.next_scan = now + self.policy['scan_every']
        blocking = any(lane != 'sidebar' or not task.name.startswith('look:')
                       for lane, task in self.runner.tasks.items())
        if (now < self.next_scan or blocking or self.ctx.camera != 'home'
                or self.ctx.last_canvas is None or snapshot.get('credits_age') is None
                or snapshot['credits_age'] > 2
                or any(snapshot.get('lanes', {}).get(lane, {}).get('status') == 'ready'
                       for lane in ('buildings', 'defenses'))):
            return False
        self.next_scan = now + self.policy['scan_every']
        cash = self.ctx.world.spendable()
        if cash is None or not self._inactive():
            return False
        buildings = sorted((b for b in default_recognizer().detect(
                            self.ctx.last_canvas, compressed=self.ctx.compressed_frames)
                            if b.side == 'ours' and b.name in self.policy['items']),
                           key=lambda b: self.policy['items'].index(b.name))
        if buildings:
            offset = self.scan_index % len(buildings)
            buildings = buildings[offset:] + buildings[:offset]
        self.scan_index += 1
        for building in buildings:
            # Entry headroom only: continuing damage can keep repairs spending.
            if cash < self.policy['min_credits']:
                continue
            point = ((building.box[0]+building.box[2])//2, (building.box[1]+building.box[3])//2)
            if any(abs(point[0]-p[0])+abs(point[1]-p[1]) < 80 for p in self.latched):
                continue
            # A sidebar observation can yield; production, placement and combat cannot.
            self.scan_active = self.runner.start('repair:'+building.name, 'sidebar', self._repair(building, point),
                                                 priority=0, preempt=True)
            return self.scan_active
        return False

    def _repair(self, building, point):
        ctx = self.ctx
        # Clear previous unit/building bars before taking the comparison image.
        yield Act([Step.click(ctx.clear_point, 'right'), Step.move(ctx.park),
                   Step.wait(self.policy['settle_ms'])])
        before = yield Look(lambda *_: ctx.last_canvas.copy())
        yield Act([Step.click(point), Step.move(ctx.park), Step.wait(self.policy['settle_ms'])])
        x0, y0, x1, _ = building.box
        region = (max(0, x0-25), max(0, y0-60), x1, y0+65)
        offsets = self.policy.get('bar_offsets', {}).get(building.name)
        diagonal = None if offsets is None else (x0+offsets[0], y0+offsets[1],
                                                  x0+offsets[2], y0+offsets[3])
        min_length = self.policy['min_known_bar_length'] if diagonal else self.policy['min_bar_length']
        min_length = self.policy.get('bar_min_lengths', {}).get(building.name, min_length)
        level = yield Look(lambda *_: selected_health(ctx.last_canvas, region, before,
                                                      min_length, diagonal))
        if level not in self.policy['levels'] or not self._inactive():
            return
        cash = ctx.world.spendable()
        if cash is None or cash < self.policy['min_credits']:
            return
        self.cleanup_pending = True
        self.cleanup_attempts = 0
        yield Act([Step.repair_mode(), Step.wait(self.policy['settle_ms'])])
        active = yield Look(lambda *_: mode_icon(ctx.last_canvas, ctx.layout, 'repair') == 'active')
        if active:
            self.latched.append(point)  # set before input, including uncertain/partial acknowledgements
            yield Act([Step.click(point), Step.wait(self.policy['settle_ms'])])
            ctx.world.note('repair-requested', item=building.name, health_band=level)
        yield from self._cleanup()
