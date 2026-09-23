"""The battlefield as tracked objects: what the inspector draws all the time, off the control loop.

Each scan (a few per second, on its own thread) finds:
- our selected units by their health bars (unitboxes.our_units): size class, health, and a name
  from the unit types we built;
- buildings by their recorded appearance, with house color determining ownership;
- unconfirmed house-color patches, which are neither unit nor building counts;
- with a trained detector (detector.py), units without health bars, by their shape.

Objects are tracked from scan to scan. When the camera moves, the whole view shifts: the shift is
measured (phase correlation of the tactical view with the previous scan's) and every track moves
with it; a jump that cannot be followed (a radar click, H) drops the tracks, since they now point
at the wrong ground. A track keeps what was learned about it: a unit named while it was selected
keeps its name after the selection ends, for as long as its house color is still seen there.
Tracks not seen lately fade, then go (TTL).
"""

from dataclasses import dataclass
import itertools
import threading
import time

import cv2
import numpy as np

from . import view as views
from .buildings import default_recognizer
from .unitboxes import our_units, recognize
from .units import enemy_blobs, house_blobs

TTL = {('ours', 'building'): 20.0, ('enemy', 'building'): 15.0, ('ours', 'named'): 8.0}
UNIT_TTL = 3.0                  # other units: seconds unseen before a track goes
FRESH = 0.35                    # seen this recently: drawn at full strength
MIN_RESPONSE = 0.3              # phase correlation peak below this: the view jumped
MESSAGES = (0, 0, 160, 64)      # game messages ("Selected across MAP") are red text up here


@dataclass
class Track:
    id: int
    box: tuple                  # (x0, y0, x1, y1) canvas pixels, x1 and y1 exclusive
    side: str                   # 'ours' or 'enemy'
    kind: str                   # 'infantry', 'vehicle', 'building', or unconfirmed 'patch'
    source: str                 # evidence: 'bars', 'color', 'template' or 'detector'
    first: float
    seen: float
    name: str | None = None
    sure: bool = False
    health: float | None = None
    hits: int = 1
    anchor: tuple | None = None  # center of the house-color patch that last confirmed it

    @property
    def center(self):
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def ttl(self):
        if self.name and self.side == 'ours':
            return TTL[('ours', 'named')]
        return TTL.get((self.side, self.kind), UNIT_TTL)

    def label(self, items=None, now=None):
        item = items.get(self.name) if items and self.name else None
        text = (item.get('short', item['name']) if item else self.name.replace('_', ' ').title() if self.name else self.kind)
        text += '' if self.sure or self.name is None else '?'
        age = None if now is None else now - self.seen
        if self.health is not None and (age is None or age < FRESH):
            text += f' {round(100 * self.health)}%'
        if age is not None and age >= 1.0:
            text += f' {age:.0f}s'
        return text


@dataclass(frozen=True)
class Seen:
    """One detection in a scan."""
    box: tuple
    side: str
    kind: str
    source: str
    name: str | None = None
    sure: bool = False
    health: float | None = None
    anchor: tuple | None = None


def _box_of(blob):
    _, _, (x, y, w, h) = blob
    return (x, y, x + w, y + h)


def _near(a, b, gap):
    return not (a[0] > b[2] + gap or b[0] > a[2] + gap or a[1] > b[3] + gap or b[1] > a[3] + gap)


class CameraShift:
    """How far the view moved since the last scan (phase correlation of the tactical view at half
    size); None when it jumped (too little in common) or there is no previous view."""

    def __init__(self, min_response=MIN_RESPONSE):
        self.min_response = min_response
        self.previous = self.window = None

    def reset(self):
        self.previous = None

    def __call__(self, canvas, region):
        x, y, w, h = region
        gray = cv2.cvtColor(canvas[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
        previous, self.previous = self.previous, small
        if previous is None or previous.shape != small.shape:
            return None
        if self.window is None or self.window.shape != small.shape:
            self.window = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(previous, small, self.window)
        if response < self.min_response:
            return None
        dx, dy = 2 * dx, 2 * dy
        return (0, 0) if abs(dx) < 1.5 and abs(dy) < 1.5 else (round(dx), round(dy))


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class Tracker:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.tracks = []
        self.ids = itertools.count(1)
        self.camera = CameraShift()

    def reset(self):
        self.tracks = []
        self.camera.reset()

    def shift(self, delta, region):
        """The view moved by delta: move every track with it; drop tracks now outside the view."""
        dx, dy = delta
        x, y, w, h = region
        kept = []
        for track in self.tracks:
            box = (track.box[0] + dx, track.box[1] + dy, track.box[2] + dx, track.box[3] + dy)
            if box[2] <= x or box[3] <= y or box[0] >= x + w or box[1] >= y + h:
                continue
            track.box = box
            if track.anchor is not None:
                track.anchor = (track.anchor[0] + dx, track.anchor[1] + dy)
            kept.append(track)
        self.tracks = kept

    def update(self, seen, now=None, delta=(0, 0), region=None):
        """One scan's detections; delta is the camera shift since the last scan (None: it jumped)."""
        now = self.clock() if now is None else now
        region = views.tactical(region)
        if delta is None:
            self.tracks = []
        elif delta != (0, 0):
            self.shift(delta, region)
        pairs = []
        for i, detection in enumerate(seen):
            for track in self.tracks:
                if track.side != detection.side or (track.kind == 'building') != (detection.kind == 'building'):
                    continue
                if track.name and detection.name and track.sure and detection.sure and track.name != detection.name:
                    continue
                overlap = iou(track.box, detection.box)
                size = min(detection.box[2] - detection.box[0], detection.box[3] - detection.box[1])
                reach = max(10.0, 0.6 * size)
                distance = float(np.hypot(track.center[0] - (detection.box[0] + detection.box[2]) / 2,
                                          track.center[1] - (detection.box[1] + detection.box[3]) / 2))
                if track.anchor is not None and detection.anchor is not None:
                    distance = min(distance, float(np.hypot(track.anchor[0] - detection.anchor[0],
                                                            track.anchor[1] - detection.anchor[1])))
                if overlap >= 0.2 or distance <= reach:
                    pairs.append((overlap - distance / 100.0, i, track))
        used, taken = set(), set()
        for _, i, track in sorted(pairs, key=lambda p: -p[0]):
            if i in used or track.id in taken:
                continue
            used.add(i)
            taken.add(track.id)
            self._merge(track, seen[i], now)
        for i, detection in enumerate(seen):
            if i not in used:
                self.tracks.append(Track(next(self.ids), detection.box, detection.side, detection.kind, detection.source,
                                         now, now, detection.name, detection.sure, detection.health,
                                         anchor=detection.anchor))
        self.tracks = [t for t in self.tracks if now - t.seen <= t.ttl]
        return self.tracks

    @staticmethod
    def _merge(track, detection, now):
        """What a detection adds to its track. A health bar is the best evidence (box, kind, name,
        health); a house-color patch only confirms where a known unit is, moving it along."""
        if detection.source == 'color' and track.source in ('bars', 'detector') and track.kind != 'building':
            if track.anchor is not None and detection.anchor is not None:
                dx, dy = detection.anchor[0] - track.anchor[0], detection.anchor[1] - track.anchor[1]
                track.box = (track.box[0] + dx, track.box[1] + dy, track.box[2] + dx, track.box[3] + dy)
        else:
            track.box = detection.box
            if detection.kind not in ('unit', 'patch') or track.kind in ('unit', 'patch', 'building'):
                track.kind = detection.kind
            track.source = detection.source
        if detection.name and (detection.sure or not track.sure):
            track.name, track.sure = detection.name, detection.sure
        if detection.health is not None:
            track.health = detection.health
        if detection.anchor is not None:
            track.anchor = detection.anchor
        track.seen = now
        track.hits += 1


def scan(canvas, region=None, built=None, items=None, detector=None, buildings=None):
    """One scan's detections: [Seen]; see the module doc."""
    region = views.tactical(region)
    buildings = default_recognizer() if buildings is None else buildings
    seen = [Seen(b.box, b.side, 'building', 'template', b.name, b.sure)
            for b in buildings.detect(canvas, region)]
    ours = recognize(our_units(canvas, region), built or {}, items or {}) if built is not None else our_units(canvas, region)
    blobs = [b for b in house_blobs(canvas, region, close=9) if b[0] >= 12 and not _in(MESSAGES, b[1])]
    free = [not any(s.side == 'ours' and _contains(s.box, b[1]) for s in seen) for b in blobs]
    for unit in ours:
        if any(s.kind == 'building' and _contains(s.box, unit.center) for s in seen):
            continue
        # A selected unit claims the house-color patch it stands on: that patch anchors its track
        # once the selection ends, and never passes for a building.
        x0, y0, x1, y1 = unit.box
        inside = [i for i, (_, center, _) in enumerate(blobs)
                  if free[i] and x0 - 4 <= center[0] <= x1 + 4 and y0 - 4 <= center[1] <= y1 + 4]
        for i in inside:
            free[i] = False
        anchor = blobs[inside[0]][1] if inside else None
        seen.append(Seen(unit.box, 'ours', unit.kind, 'bars', unit.name, unit.sure, unit.health, anchor))
    patches = [(_box_of(b), 'ours', b[1]) for b, f in zip(blobs, free) if f]
    patches += [(_box_of(b), 'enemy', b[1]) for b in enemy_blobs(canvas, region)
                if not _in(MESSAGES, b[1]) and not any(s.side == 'enemy' and _contains(s.box, b[1]) for s in seen)]
    claimed = set()
    if detector is not None:
        for found in detector.detect(canvas, region):
            if any(iou(found.box, s.box) >= 0.3 or
                   (s.kind == 'building' and _contains(s.box, ((found.box[0] + found.box[2]) / 2,
                                                                            (found.box[1] + found.box[3]) / 2)))
                   for s in seen):
                continue
            inside = [i for i, (box, side, center) in enumerate(patches)
                      if side == found.side and _contains(found.box, center)]
            claimed.update(inside)
            seen.append(Seen(found.box, found.side, found.kind, 'detector', anchor=patches[inside[0]][2] if inside else None))
    for i, (box, side, center) in enumerate(patches):
        if i not in claimed:
            seen.append(Seen(box, side, 'patch', 'color', anchor=center))
    return seen


def _contains(box, point):
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


def _in(box, point):
    x, y, w, h = box
    return x <= point[0] < x + w and y <= point[1] < y + h


class SceneWorker:
    """Scans the latest submitted view on its own thread (the control loop only hands frames over),
    at most every `period` seconds, and publishes the tracks and the scan's cost."""

    def __init__(self, built=lambda: None, items=None, detector=None, period=0.2, on_scan=None,
                 clock=time.monotonic):
        self.built, self.items, self.detector = built, items, detector
        self.period, self.on_scan, self.clock = period, on_scan, clock
        self.tracker = Tracker(clock)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.latest = None               # (canvas, region) or None: nothing to scan (menus)
        self.tracks = []
        self.stats = dict(scans=0, ms=None, tracks=0, shift=None)
        self.error = None
        self.stopped = False
        self.thread = threading.Thread(target=self._run, name='scene', daemon=True)
        self.thread.start()

    def submit(self, canvas, region):
        with self.lock:
            self.latest = (canvas, tuple(region))
        self.wake.set()

    def clear(self):
        """Not in a match: forget the tracks."""
        with self.lock:
            self.latest = None
            self.tracker.reset()
            self.tracks = []

    def snapshot(self):
        with self.lock:
            return list(self.tracks), dict(self.stats)

    def step(self, canvas, region):
        """One scan now, on the calling thread (tests and tools)."""
        started = time.perf_counter()
        seen = scan(canvas, region, self.built(), self.items, self.detector)
        delta = self.tracker.camera(canvas, region)
        tracks = self.tracker.update(seen, self.clock(), delta, region)
        with self.lock:
            self.tracks = [Track(**vars(t)) for t in tracks]
            self.stats = dict(scans=self.stats['scans'] + 1, ms=round((time.perf_counter() - started) * 1000, 1),
                              tracks=len(tracks), shift=delta)
        if self.on_scan is not None:
            self.on_scan(canvas, self.tracks)
        return self.tracks

    def _run(self):
        last = 0.0
        while not self.stopped:
            self.wake.wait(0.5)
            self.wake.clear()
            if self.stopped:
                break
            wait = self.period - (self.clock() - last)
            if wait > 0:
                time.sleep(wait)
            with self.lock:
                latest = self.latest
            if latest is None:
                continue
            last = self.clock()
            try:
                self.step(*latest)
                self.error = None
            except Exception as error:          # a bad frame must not end the inspector
                self.error = f'{type(error).__name__}: {error}'

    def close(self):
        self.stopped = True
        self.wake.set()
        self.thread.join(2)


def confirmed(track):
    """Object evidence, never a class guessed from a patch's area."""
    return track.source in ('bars', 'detector', 'template')


def draw_scene(image, tracks, items=None, now=None, scale=1.0, font=0.34):
    """Tracks for the inspector: ours green (yellow when the name is a guess, teal for buildings),
    the enemy's red (orange for buildings); boxes fade as their tracks go unseen. Every box is
    drawn; a label only where it covers no other, and only for what is known to be a unit or a
    building (a lone house-color patch gets a thin box)."""
    now = time.monotonic() if now is None else now
    taken = []
    canvas_h, canvas_w = image.shape[:2]
    for track in sorted(tracks, key=lambda t: (t.kind != 'building', t.box[1], t.box[0])):
        age = now - track.seen
        if track.side == 'ours':
            color = (200, 200, 0) if track.kind == 'building' else \
                (0, 200, 0) if track.sure or track.name is None else (0, 210, 230)
            if not confirmed(track):
                color = (120, 220, 120)
        else:
            color = (0, 90, 230) if track.kind == 'building' else (0, 0, 230)
        fade = 1.0 if age < FRESH else max(0.25, 1.0 - age / max(track.ttl, 1.0))
        color = tuple(int(c * fade + 60 * (1 - fade)) for c in color)
        known = confirmed(track)
        if not known:
            color = tuple(round(c * 0.55) for c in color)
        x0, y0, x1, y1 = (int(round(v * scale)) for v in track.box)
        if x1 <= 0 or y1 <= 0 or x0 >= canvas_w or y0 >= canvas_h:
            continue
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(canvas_w, x1), min(canvas_h, y1)
        # One display pixel, with no heavy black outer stroke obscuring small sprites.
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 1, cv2.LINE_AA)
        if not known:
            continue
        text = track.label(items, now)
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
        width, height = width + 6, height + baseline + 5
        if width > canvas_w or height > canvas_h:
            continue
        left = min(x0, canvas_w - width)
        candidates = [(left, max(0, min(top, canvas_h - height)), left + width,
                       max(0, min(top, canvas_h - height)) + height) for top in (y0 - height - 2, y1 + 2)]
        area = next((a for a in candidates if not any(_near(a, t, 1) for t in taken)), None)
        if area is None:
            continue
        taken.append(area)
        left, top, right, bottom = area
        plate = image[top:bottom, left:right]
        plate[:] = (plate.astype(np.float32) * 0.25 + np.array((22, 26, 30)) * 0.75).astype(np.uint8)
        text_color = tuple(round((130 + c * 0.5) * fade) for c in color)
        cv2.putText(image, text, (left + 3, bottom - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font, text_color, 1, cv2.LINE_AA)
    return image


def summary(tracks, items=None, now=None):
    """Counts for the inspector's text panel: ours by label, the enemy's by kind; fresh tracks only."""
    now = time.monotonic() if now is None else now
    fresh = [t for t in tracks if now - t.seen < 1.0]
    ours = {}
    for track in fresh:
        if track.side == 'ours' and confirmed(track):
            key = track.label(items).rsplit(' ', 1)[0] if track.health is not None else track.label(items)
            ours[key] = ours.get(key, 0) + 1
    health = [t.health for t in fresh if t.side == 'ours' and confirmed(t) and t.health is not None]
    return dict(ours=ours, health=round(sum(health) / len(health), 2) if health else None,
                patches=sum(1 for t in fresh if t.side == 'ours' and not confirmed(t)),
                enemy_patches=sum(1 for t in fresh if t.side == 'enemy' and not confirmed(t)),
                enemy=dict(unit=sum(1 for t in fresh if t.side == 'enemy' and confirmed(t) and t.kind != 'building'),
                           building=sum(1 for t in fresh if t.side == 'enemy' and confirmed(t) and t.kind == 'building')),
                tracked=len(tracks))
