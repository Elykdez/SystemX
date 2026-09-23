"""Compact, persistent battlefield description with ages, commitments, and a summarizer.

Facts carry the time they were observed. Our own base comes mostly from the action
ledger (what we queued and placed, once confirmed), then from sidebar evidence.
Nothing here is sent to System One directly: the reflex words the facts each question needs, in
the words of the System One profile (playbook.py).
"""

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time

LANES = ('buildings', 'defenses', 'infantry', 'vehicles')
PROFILES = Path(__file__).with_name('profiles')


def load_knowledge(path=PROFILES / 'knowledge-ra2-soviet.json'):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('version') != 1:
        raise ValueError('Unsupported knowledge file.')
    return data


@dataclass
class Fact:
    value: object = None
    at: float = 0.0

    def age(self, now):
        return None if not self.at else now - self.at

    def fresh(self, now, seconds):
        return self.at and now - self.at <= seconds


@dataclass
class Slot:
    index: int
    item: str | None
    state: str                 # empty | available | disabled | building | ready | hold | unknown
    progress: float | None = None
    count: int | None = None


@dataclass
class LaneState:
    status: str = 'unknown'    # unknown | idle | building | ready | hold
    item: str | None = None
    progress: float | None = None
    available: tuple = ()
    at: float = 0.0


@dataclass
class Commitment:
    id: int
    kind: str                  # queue | place | order | deploy
    lane: str
    item: str | None
    cost: int = 0
    status: str = 'requested'  # requested | confirmed | done | failed | cancelled
    created: float = field(default_factory=time.monotonic)
    detail: dict = field(default_factory=dict)


class Ledger:
    """Requested actions and credit reservations, so two tasks never spend the same money."""

    def __init__(self):
        self.items, self.next_id = {}, 1

    def add(self, kind, lane, item=None, cost=0, **detail):
        entry = Commitment(self.next_id, kind, lane, item, cost, detail=detail)
        self.items[entry.id] = entry
        self.next_id += 1
        return entry

    def open(self, kind=None, lane=None):
        return [c for c in self.items.values() if c.status in ('requested', 'confirmed')
                and (kind is None or c.kind == kind) and (lane is None or c.lane == lane)]

    def reserved(self):
        return sum(c.cost for c in self.open('queue') if c.status == 'requested')

    def close(self, entry, status):
        entry.status = status

    def prune(self, keep=400):
        if len(self.items) > keep:
            for key in sorted(self.items)[:len(self.items) - keep]:
                if self.items[key].status not in ('requested', 'confirmed'):
                    del self.items[key]


class World:
    def __init__(self, knowledge=None, clock=time.monotonic, log=None):
        self.knowledge = knowledge or load_knowledge()
        self.clock = clock
        self.log = log                 # optional run log: world notes (orders, sightings) are written there too
        self.lock = threading.RLock()
        self.started = clock()
        self.mode = Fact('unknown')
        self.credits = Fact()
        self.power = Fact()            # dict(level='ok'|'low'|'unknown', fill=0..1)
        self.active_tab = Fact()
        self.lanes = {lane: LaneState() for lane in LANES}
        self.slots = {lane: Fact([]) for lane in LANES}
        self.radar = Fact()            # perception.radar.Radar: map outline, camera box, own and enemy dots
        self.ledger = Ledger()
        self.structures = {}           # item -> confirmed count (placed and seen)
        self.units_built = {}          # item -> count confirmed produced
        self.groups = {}               # group id -> dict(size, order, target, since)
        self.events = deque(maxlen=64)
        self.mcv_deployed = Fact(False)
        self.sidebar_empty = Fact(False)
        self.home = Fact()             # canvas point of our construction yard at camera home
        self.camera = Fact('home')     # named camera anchor believed to be in view
        self._refinery_unlocked = False
        self._refinery_missing = []
        self._radar_missing = {}
        self._technology_missing = {}
        self._factory_missing = []
        self._last_placement = float('-inf')
        self.disabled_lanes = set()
        self._tab_missing = {}
        self._lane_busy_at = {}

    # -- updates from perception ----------------------------------------
    def note(self, kind, **detail):
        with self.lock:
            if kind in ('placed', 'deployed'):
                self._last_placement = self.clock()
                self._refinery_missing.clear()
            self.events.append(dict(t=round(self.clock() - self.started, 1), kind=kind, **detail))
        if self.log is not None and kind != 'lane':
            self.log(dict(time=time.time(), world=kind, **detail))

    def update_sidebar(self, lane, slots, now=None):
        now = now or self.clock()
        with self.lock:
            self.slots[lane] = Fact(slots, now)
            self.active_tab = Fact(lane, now)
            building = next((s for s in slots if s.state in ('building', 'ready', 'hold')), None)
            # A unit with a queue count is waiting (or its build clock is too early to see): busy.
            queued = next((s for s in slots if s.count and s.item), None)
            available = tuple(s.item for s in slots if s.item and s.state in ('available', 'building', 'ready', 'hold'))
            state = self.lanes[lane]
            previous = (state.status, state.item)
            if building is not None:
                state.status, state.item, state.progress = building.state, building.item, building.progress
                self._lane_busy_at[lane] = now
            elif queued is not None:
                state.status, state.item, state.progress = 'building', queued.item, None
                self._lane_busy_at[lane] = now
            elif (lane in ('buildings', 'defenses') and state.status == 'building' and
                  now - self._lane_busy_at.get(lane, float('-inf')) <= 2.0 and
                  any(s.item == state.item and s.state == 'available' for s in slots)):
                # The clock disappears a frame before Ready; do not click the cameo again.
                pass
            else:
                state.status, state.item, state.progress = 'idle', None, None
            state.available, state.at = available, now
            if previous != (state.status, state.item) and previous[0] != 'unknown':
                self.note('lane', lane=lane, status=state.status, item=state.item)

    def reconcile_tabs(self, states, now=None):
        """Repeated blank production buttons prove that the last producer for that lane is gone."""
        now = self.clock() if now is None else now
        producers = {'buildings': 'construction_yard', 'infantry': 'barracks', 'vehicles': 'war_factory'}
        with self.lock:
            for lane in LANES:
                enabled = states.get(lane)
                if enabled is not False:
                    self._tab_missing.pop(lane, None)
                    if enabled is True:
                        self.disabled_lanes.discard(lane)
                    continue
                samples = self._tab_missing.setdefault(lane, [])
                if samples and now - samples[-1] > 3:
                    samples.clear()
                if not samples or now > samples[-1]:
                    samples.append(now)
                if len(samples) < 3 or now - samples[0] < 2:
                    continue
                self.disabled_lanes.add(lane)
                self.lanes[lane] = LaneState(status='unknown', at=now)
                self.slots[lane] = Fact([], now)
                for entry in self.ledger.open(lane=lane):
                    self.ledger.close(entry, 'cancelled')
                item = producers.get(lane)
                count = self.structures.get(item, 0)
                if count:
                    self.structures[item] = 0
                    self.note('structure-lost', item=item, previous=count, remaining=0,
                              evidence='production tab repeatedly matches its disabled button')
                # Keep the confirmation bounded while the lane remains disabled.
                self._tab_missing[lane] = [samples[0], now]

    def reconcile_structures(self, lane, slots, complete=False, now=None):
        """Infer structure losses only from stable, complete sidebar evidence.

        A complete top-of-list Soviet sidebar must previously have shown both refinery-gated
        options. Losing either option alone, switching tabs, or seeing fewer colored objects
        says nothing about refinery counts. Repeated full readings can establish zero, not how
        many refineries survived when at least one remains.
        """
        now = self.clock() if now is None else now
        with self.lock:
            self.reconcile_radar(lane, slots, complete, now)
            self.reconcile_technology(lane, slots, complete, now)
            self.reconcile_factory(lane, slots, complete, now)
            if lane is not None and lane != 'buildings':
                return
            known = ('available', 'disabled', 'building', 'ready', 'hold')
            readable = (complete and bool(slots) and slots[-1].state == 'empty'
                        and all((s.item is None and s.state == 'empty') or
                                (s.item in self.knowledge['items'] and s.state in known) for s in slots))
            items = {s.item for s in slots if s.item}
            gates = set(self.knowledge.get('refinery_gates', ['war_factory', 'radar_tower']))
            if lane != 'buildings' or not readable:
                self._refinery_missing.clear()
                return
            if gates <= items:
                self._refinery_unlocked = True
            count = self.structures.get('ore_refinery', 0)
            anchors = [s.item for s in slots[:3]] == self.knowledge.get(
                'opening_anchors', ['tesla_reactor', 'ore_refinery', 'barracks'])
            busy = any(s.state in ('building', 'ready', 'hold') or s.count for s in slots)
            pending = any(c.status == 'requested' for c in self.ledger.open(lane='buildings'))
            if (not self._refinery_unlocked or not count or not anchors or gates & items or busy or pending
                    or now - self._last_placement < 3.0):
                self._refinery_missing.clear()
                return
            previous = self._refinery_missing
            if previous and (now - previous[-1][0] > 10.0 or count != previous[-1][1]):
                previous.clear()
            if not previous or now > previous[-1][0]:
                previous.append((now, count))
            if len(previous) >= 3 and now - previous[0][0] >= 2.0:
                self.structures['ore_refinery'] = 0
                self._refinery_unlocked = False
                previous.clear()
                self.note('structure-lost', item='ore_refinery', previous=count, remaining=0,
                          evidence='refinery prerequisites disappeared from complete buildings sidebar')

    def reconcile_factory(self, lane, slots, complete, now):
        """Aircraft can keep the Allied vehicle tab enabled after all ground factories are lost."""
        plan = self.knowledge.get('factory_loss')
        if not plan or not self.has('war_factory'):
            self._factory_missing.clear()
            return
        if lane not in (None, 'vehicles'):
            return
        items = {s.item for s in slots if s.item}
        survivors = set(plan['survivors'])
        readable = (lane == 'vehicles' and complete and slots and slots[-1].state == 'empty'
                    and slots[0].item in survivors and items and items <= survivors
                    and all(s.state in ('available', 'disabled', 'building', 'hold', 'empty') for s in slots))
        if not readable or now - self._last_placement < 3:
            self._factory_missing.clear()
            return
        samples = self._factory_missing
        if samples and now - samples[-1] > 10:
            samples.clear()
        if not samples or now > samples[-1]:
            samples.append(now)
        if len(samples) < 3 or now - samples[0] < 2:
            return
        count = self.structures['war_factory']
        self.structures['war_factory'] = 0
        for entry in self.ledger.open(lane='vehicles'):
            if entry.item not in survivors:
                self.ledger.close(entry, 'cancelled')
        samples.clear()
        self.note('structure-lost', item='war_factory', previous=count, remaining=0,
                  evidence='complete vehicle sidebar repeatedly contains only aircraft from another producer')

    def reconcile_radar(self, lane, slots, complete, now):
        plan = self.knowledge.get('radar_loss')
        if not plan:
            return
        radar = self.radar.value
        if (not self.has(plan['building']) or radar is None or radar.online or
                self.radar.age(now) > 2 or (self.power.value or {}).get('level') != 'ok' or
                now - self._last_placement < 3):
            self._radar_missing.clear()
            return
        self._reconcile_unlock_loss(lane, slots, complete, now, plan, self._radar_missing,
                                    'powered radar offline and exclusive options absent from two complete lanes')

    def reconcile_technology(self, lane, slots, complete, now):
        plan = self.knowledge.get('technology_loss')
        if not plan:
            return
        radar = self.radar.value
        if (not self.has(plan['building']) or radar is None or not radar.online or
                self.radar.age(now) > 2 or (self.power.value or {}).get('level') != 'ok' or
                now - self._last_placement < 3):
            self._technology_missing.clear()
            return
        self._reconcile_unlock_loss(lane, slots, complete, now, plan, self._technology_missing,
                                    'powered radar online and exclusive options absent from two complete lanes')

    def _reconcile_unlock_loss(self, lane, slots, complete, now, plan, missing, evidence):
        if lane not in plan['gates']:
            return
        readable = complete and slots and slots[-1].state == 'empty' and all(
            s.state in ('available', 'disabled', 'building', 'ready', 'hold', 'empty') for s in slots)
        anchors = plan['anchors'][lane]
        if (not readable or [s.item for s in slots[:len(anchors)]] != anchors or
                set(plan['gates'][lane]) & {s.item for s in slots}):
            missing.pop(lane, None)
            return
        samples = missing.setdefault(lane, [])
        if samples and now - samples[-1] > 10:
            samples.clear()
        if not samples or now > samples[-1]:
            samples.append(now)
        if len(samples) > 3:
            missing[lane] = [samples[0], *samples[-2:]]
        if all(len(missing.get(name, [])) >= 3 and
               missing[name][-1] - missing[name][0] >= 2 and
               now - missing[name][-1] <= 10 for name in plan['gates']):
            item = plan['building']
            count = self.structures[item]
            self.structures[item] = 0
            missing.clear()
            self.note('structure-lost', item=item, previous=count, remaining=0,
                      evidence=evidence)

    def set(self, name, value, now=None):
        with self.lock:
            setattr(self, name, Fact(value, now or self.clock()))

    # -- derived views ----------------------------------------------------
    def spendable(self):
        with self.lock:
            credits = self.credits.value
            return None if credits is None else credits - self.ledger.reserved()

    def item(self, item_id):
        return self.knowledge['items'][item_id]

    def has(self, item_id):
        return self.structures.get(item_id, 0) > 0

    def power_margin(self):
        """Estimated from confirmed structures when the power bar is unreadable."""
        items = self.knowledge['items']
        total = sum(items[i].get('power', 0) * n for i, n in self.structures.items() if i in items)
        return total

    def snapshot(self):
        """Plain-data copy for expectation checks and logs."""
        with self.lock:
            now = self.clock()
            return dict(
                t=round(now - self.started, 2), mode=self.mode.value, credits=self.credits.value,
                credits_age=self.credits.age(now), power=self.power.value, tab=self.active_tab.value,
                lanes={lane: dict(status=s.status, item=s.item, progress=s.progress,
                                  available=list(s.available), age=None if not s.at else round(now - s.at, 2))
                       for lane, s in self.lanes.items()},
                structures=dict(self.structures), units=dict(self.units_built),
                disabled_lanes=sorted(self.disabled_lanes),
                mcv_deployed=self.mcv_deployed.value, sidebar_empty=bool(self.sidebar_empty.value),
                groups={k: dict(v) for k, v in self.groups.items()},
                reserved=self.ledger.reserved(),
                radar=self.radar.value.summary() if self.radar.value is not None else None)
