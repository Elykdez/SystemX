"""Only complete, repeated prerequisite evidence can clear the last refinery count."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from addons.chronodivide import skills
from addons.chronodivide.layout import Layout, PROFILES
from addons.chronodivide.perception import reader
from addons.chronodivide.perception.sidebar import CameoCatalog, lane_of, read_slots
from addons.chronodivide.reflex import Reflex
from addons.chronodivide.world import Slot, World

LOCKED = ('tesla_reactor', 'ore_refinery', 'barracks', 'service_depot', 'battle_lab')
UNLOCKED = LOCKED + ('war_factory', 'radar_tower')
FIXTURES = Path(__file__).with_name('fixtures')


def slots(items=LOCKED):
    return [Slot(i, item, 'available') if item else Slot(i, None, 'empty')
            for i, item in enumerate(list(items) + [None] * (20 - len(items)))]


def world(unlocked=True):
    result = World(clock=lambda: 100.0)
    result.structures.update(construction_yard=1, tesla_reactor=2, barracks=1,
                             ore_refinery=2, war_factory=1, radar_tower=1)
    result.set('credits', 5000)
    result.set('power', {'level': 'ok'})
    if unlocked:
        result.reconcile_structures('buildings', slots(UNLOCKED), complete=True, now=90)
    return result


def observe(w, at, readings=None, complete=True, lane='buildings'):
    w.reconcile_structures(lane, slots() if readings is None else readings, complete, at)


def test_last_refinery_loss_clears_stale_count_and_existing_reflex_rebuilds():
    w = world()
    observe(w, 100)
    observe(w, 101)
    assert w.structures['ore_refinery'] == 2
    observe(w, 102)
    assert w.structures['ore_refinery'] == 0
    assert w.structures['war_factory'] == w.structures['radar_tower'] == 1
    w.update_sidebar('buildings', slots(), now=102)
    reflex = Reflex(skills.Context(Layout.for_canvas(1024, 768), w))
    assert reflex.plan(w.snapshot(), 102)[0] == 'queue:ore_refinery'
    assert [event['kind'] for event in w.events] == ['structure-lost']


@pytest.mark.parametrize('complete', [False, True])
def test_no_previously_observed_unlock_means_no_loss_inference(complete):
    w = world(unlocked=False)
    for now in (100, 101, 103, 110):
        observe(w, now, complete=complete)
    assert w.structures['ore_refinery'] == 2


@pytest.mark.parametrize('bad', ['unknown', 'menu', 'building', 'ready', 'hold', 'partial', 'scrolled'])
def test_incomplete_or_transient_readings_reset_confirmation(bad):
    w = world()
    observe(w, 100)
    observe(w, 101)
    readings = slots()
    if bad == 'partial':
        complete = False
    else:
        complete = True
        if bad == 'scrolled':
            readings[0].item = 'service_depot'
        else:
            readings[4].state = bad
            if bad in ('unknown', 'menu'):
                readings[4].item = None
    observe(w, 102, readings, complete)
    observe(w, 103)
    observe(w, 104)
    assert w.structures['ore_refinery'] == 2
    observe(w, 105)
    assert w.structures['ore_refinery'] == 0


@pytest.mark.parametrize('gate', ['war_factory', 'radar_tower'])
def test_either_gate_still_present_even_disabled_preserves_refineries(gate):
    w = world()
    readings = slots(LOCKED + (gate,))
    readings[5].state = 'disabled'
    for now in (100, 101, 103):
        observe(w, now, readings)
    assert w.structures['ore_refinery'] == 2


def test_distinct_observations_must_span_two_seconds_and_not_a_long_gap():
    w = world()
    for now in (100, 100, 100, 100.1, 100.2):
        observe(w, now)
    assert w.structures['ore_refinery'] == 2
    observe(w, 120)
    observe(w, 121)
    assert w.structures['ore_refinery'] == 2
    observe(w, 122)
    assert w.structures['ore_refinery'] == 0


def test_tab_visits_preserve_evidence_but_unreadable_frame_resets_it():
    w = world()
    observe(w, 100)
    observe(w, 100.5, slots(('conscript',)), lane='infantry')
    observe(w, 101)
    observe(w, 102, [], complete=False, lane=None)
    observe(w, 103)
    assert w.structures['ore_refinery'] == 2
    observe(w, 104, slots(('rhino_tank',)), lane='vehicles')
    observe(w, 105)
    observe(w, 106)
    assert w.structures['ore_refinery'] == 0


def test_new_placement_count_changes_and_pending_actions_cannot_erase_new_building():
    w = world()
    observe(w, 100)
    observe(w, 101)
    w.structures['ore_refinery'] += 1
    observe(w, 102)
    assert w.structures['ore_refinery'] == 3
    w.note('placed', item='ore_refinery')
    for now in (100, 101, 102):
        observe(w, now)
    assert w.structures['ore_refinery'] == 3
    action = w.ledger.add('queue', 'buildings', 'ore_refinery', 2000)
    for now in (104, 105, 106):
        observe(w, now)
    assert w.structures['ore_refinery'] == 3
    w.ledger.close(action, 'cancelled')


def test_loss_cannot_repeat_after_rebuild_without_fresh_unlock_evidence():
    w = world()
    for now in (100, 101, 102):
        observe(w, now)
    w.structures['ore_refinery'] = 1
    for now in (110, 111, 112):
        observe(w, now)
    assert w.structures['ore_refinery'] == 1
    assert len(w.events) == 1


def test_reader_only_marks_full_grid_readings_complete(monkeypatch):
    layout = Layout.for_canvas(1024, 768)
    mode = {'mode': 'hud'}
    readings = [SimpleNamespace(item=s.item, state=s.state, progress=None, queued=False) for s in slots()]
    monkeypatch.setattr(reader, 'read_slots', lambda *args: readings)
    monkeypatch.setattr(reader, 'read_power', lambda *args: {})
    monkeypatch.setattr(reader, 'read_radar', lambda *args: None)
    p = reader.Perception.__new__(reader.Perception)
    p.layout, p.compressed, p.catalog = layout, False, None
    p.blank_tabs = None
    p.profile = SimpleNamespace(raw={})
    p.modes = SimpleNamespace(read=lambda canvas: mode)
    p.credits = SimpleNamespace(read=lambda crop: None)
    w = world()
    canvas = np.zeros((768, 1024, 3), np.uint8)
    p.read(canvas, w, 100)
    readings[4].item, readings[4].state = None, 'unknown'
    p.read(canvas, w, 101)
    readings[4].item, readings[4].state = 'battle_lab', 'available'
    for now in (102, 103):
        p.read(canvas, w, now)
    assert w.structures['ore_refinery'] == 2
    p.read(canvas, w, 104)
    assert w.structures['ore_refinery'] == 0


def test_recorded_browser_sidebar_confirms_refinery_loss_and_requests_rebuild():
    # Sidebar from frame-1790577493285694300 in the 20260928-142727-903582 run.
    # Both refineries were gone; the existing factory and radar still stood. The Battle Lab
    # cameo had to be taught before this became a complete observation rather than an unknown.
    canvas = np.zeros((768, 1024, 3), np.uint8)
    canvas[:, 850:] = cv2.imread(str(FIXTURES / 'refinery-loss-sidebar-browser.png'))
    catalog = CameoCatalog.load(PROFILES / 'cameos-1024x768.npz', compressed=True)
    catalog.teach('battle_lab', cv2.imread(str(FIXTURES / 'battle-lab-browser.png')))
    layout = Layout.for_canvas(1024, 768)
    readings = read_slots(canvas, layout, catalog)
    assert [r.item for r in readings if r.item] == list(LOCKED)
    assert all(r.state in ('available', 'empty') for r in readings)
    lane = lane_of(readings, World().knowledge)
    observed = [Slot(i, r.item, r.state, r.progress, 1 if r.queued else None)
                for i, r in enumerate(readings)]
    w = world()
    for now in (100, 101, 102):
        w.update_sidebar(lane, observed, now)
        w.reconcile_structures(lane, observed, complete=True, now=now)
    assert w.structures['ore_refinery'] == 0
    reflex = Reflex(skills.Context(layout, w))
    assert reflex.plan(w.snapshot(), 102)[0] == 'queue:ore_refinery'
