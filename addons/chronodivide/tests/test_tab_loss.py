from pathlib import Path

import cv2
import numpy as np

from addons.chronodivide.layout import Layout
from addons.chronodivide.perception.reader import Perception
from addons.chronodivide.perception.modes import ModeReader
from addons.chronodivide.perception.tabs import read_tabs
from addons.chronodivide.tests.test_policy import setup
from addons.chronodivide.world import Slot, World


FIXTURES = Path(__file__).with_name('fixtures')
LAYOUT = Layout.for_canvas(1024, 768)


def test_recorded_disabled_buttons_are_distinct_from_enabled_and_unknown_views():
    enabled = cv2.imread(str(FIXTURES / 'tabs-enabled-browser.png'))
    lost = cv2.imread(str(FIXTURES / 'tabs-lost-browser.png'))
    assert read_tabs(enabled, LAYOUT) == dict.fromkeys(LAYOUT.tabs, True)
    assert read_tabs(lost, LAYOUT) == dict(buildings=False, defenses=False, infantry=False, vehicles=True)
    assert read_tabs(np.full_like(lost, 170), LAYOUT) == dict.fromkeys(LAYOUT.tabs)


def test_repeated_recorded_loss_clears_producers_and_stale_lanes_not_unrelated_buildings():
    w = World(clock=lambda: 100)
    w.structures.update(construction_yard=1, barracks=1, war_factory=1, ore_refinery=2, tesla_reactor=2)
    w.set('mcv_deployed', True)
    w.update_sidebar('buildings', [Slot(0, 'ore_refinery', 'available')])
    queued = w.ledger.add('queue', 'infantry', 'conscript', cost=100)
    frame = cv2.imread(str(FIXTURES / 'tabs-lost-browser.png'))
    reference = cv2.imread(str(FIXTURES / 'tabs-enabled-browser.png'))
    perception = Perception(modes=ModeReader.from_frame(reference, compressed=True), compressed=True)
    for now in (100, 101):
        assert perception.read(frame, w, now)[0]['mode'] == 'hud'
        assert w.has('construction_yard') and w.has('barracks')
    perception.read(frame, w, 102)
    assert not w.has('construction_yard') and not w.has('barracks')
    assert w.structures['ore_refinery'] == 2 and w.structures['tesla_reactor'] == 2 and w.has('war_factory')
    assert w.disabled_lanes == {'buildings', 'defenses', 'infantry'}
    assert not w.lanes['buildings'].available and queued.status == 'cancelled'
    assert w.mcv_deployed.value  # loss must not re-enable the opening deploy shortcut


def test_unknown_frames_break_confirmation_and_enabled_buttons_restore_lane():
    w = World(clock=lambda: 100)
    w.structures['war_factory'] = 1
    w.reconcile_tabs({'vehicles': False}, 100)
    w.reconcile_tabs({}, 101)
    w.reconcile_tabs({'vehicles': False}, 102)
    w.reconcile_tabs({'vehicles': False}, 103)
    assert w.has('war_factory')
    w.reconcile_tabs({'vehicles': False}, 104)
    assert not w.has('war_factory') and 'vehicles' in w.disabled_lanes
    w.reconcile_tabs({'vehicles': True}, 105)
    assert 'vehicles' not in w.disabled_lanes


def test_disabled_defense_tab_does_not_delete_barracks_or_schedule_tab_retry():
    w, ctx, runner, reflex, executor, time = setup()
    w.structures.update(barracks=1, war_factory=1, ore_refinery=1)
    w.set('mcv_deployed', True)
    w.units_built['rhino_tank'] = 4
    w.set('credits', 5000)
    for now in (98, 99, 100):
        w.reconcile_tabs({'defenses': False, 'infantry': True, 'vehicles': True}, now)
    assert w.has('barracks')
    reflex.army.threat_at = time[0]
    w.update_sidebar('vehicles', [Slot(0, 'rhino_tank', 'building')])
    w.update_sidebar('infantry', [Slot(0, 'conscript', 'building')])
    assert reflex.plan(w.snapshot(), time[0]) is None
