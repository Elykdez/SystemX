"""A gem order needs a fresh field, a model choice and a rechecked miner."""

from dataclasses import replace

import numpy as np
import pytest

from addons.chronodivide import miners
from addons.chronodivide.perception.unitboxes import Unit
from addons.chronodivide.tests.test_miners import drive
from addons.chronodivide.tests.test_playbook import game
from addons.chronodivide.tests.test_strategy import CameraGame, online_radar, run, setup


def vehicle(health=1):
    return Unit((390, 390, 420, 420), 'ours', 'vehicle', health)


def ready():
    world, army, reflex, time = game()
    world.set('credits', 700)
    state, plan = army.miners, army.book['resources']
    state.grouped = True
    state.home_ore_seen = False
    state.home_ore_at = time[0]
    state.home_ore_thin_since = time[0] - plan['home_thin_seconds']
    state.field, state.field_at, state.field_chips = (920, 80), time[0], 15
    return world, army, reflex, time, state, plan


def test_model_rule_chooses_gems_only_for_thin_ore_low_cash_and_quiet_base():
    world, army, reflex, time, state, plan = ready()
    assert reflex.feasible_resources(time[0]) == ['keep_home', 'harvest_gems']
    words, facts = reflex.resource_facts(time[0])
    assert 'gem field' in words['gem_field']
    assert reflex.book.rule('resources', reflex.feasible_resources(time[0]), facts) == 'harvest_gems'

    for change in ('home_ore', 'money', 'base_threat'):
        world, army, reflex, time, state, plan = ready()
        if change == 'home_ore':
            state.home_ore_seen, state.home_ore_thin_since = True, None
        elif change == 'money':
            world.set('credits', plan['low_credits_below'] + 1)
        else:
            army.threat_at = time[0] - 1
        assert reflex.feasible_resources(time[0]) == ['keep_home']
        assert reflex.book.rule('resources', reflex.feasible_resources(time[0]),
                                reflex.resource_facts(time[0])[1]) == 'keep_home'


@pytest.mark.parametrize('missing', ['field', 'team', 'home', 'campaign', 'cooldown'])
def test_stale_or_unavailable_evidence_removes_relocation_option(missing):
    world, army, reflex, time, state, plan = ready()
    if missing == 'field':
        state.field_at -= plan['field_fresh_within'] + 1
    elif missing == 'team':
        state.grouped = False
    elif missing == 'home':
        state.home_ore_at -= plan['home_fresh_within'] + 1
    elif missing == 'campaign':
        army.campaign.active = True
        army.campaign.started = time[0]
    else:
        state.resource_last = time[0]
    assert reflex.feasible_resources(time[0]) == ['keep_home']
    army.intent.resources = 'harvest_gems'
    reflex.ask_strategy(time[0])
    assert army.intent.resources == 'keep_home'


def test_home_ore_needs_repeated_scarcity_and_recent_sample(monkeypatch):
    world, army, reflex, time, state, plan = ready()
    state.home_ore_thin_since = None
    state.home_ore_at = -1e9
    army.ctx.camera = 'home'
    army.ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    monkeypatch.setattr(miners, 'ore_supply', lambda *args: plan['home_min_chips'] - 1)
    miners.observe_home(army.ctx, state, plan, time[0])
    assert not miners.home_thin(state, plan, time[0])
    time[0] += plan['home_thin_seconds']
    miners.observe_home(army.ctx, state, plan, time[0])
    assert miners.home_thin(state, plan, time[0])
    time[0] += plan['home_fresh_within'] + 1
    assert not miners.home_thin(state, plan, time[0])


def test_model_can_survey_before_a_field_is_known_and_requires_explored_ground():
    world, army, reflex, time, state, plan = ready()
    state.field = None
    radar = online_radar()
    world.set('radar', replace(radar, enemy=(), explored=np.ones_like(radar.explored)))
    army.ctx.radar_home = radar.own[0][1]
    options = reflex.feasible_resources(time[0])
    assert options == ['keep_home', 'survey_gems']
    assert reflex.book.rule('resources', options, reflex.resource_facts(time[0])[1]) == 'survey_gems'
    army.intent.resources = 'survey_gems'
    assert miners.survey_due(army, time[0])

    world.set('radar', replace(radar, enemy=(), explored=np.zeros_like(radar.explored)))
    assert miners.survey_point(army) is None
    assert not miners.survey_due(army, time[0])
    assert reflex.feasible_resources(time[0]) == ['keep_home']


def test_resource_survey_observes_fresh_crystals_without_commanding_units(monkeypatch):
    world, ctx, runner, army, reflex, executor, time = setup()
    radar = online_radar()
    world.set('radar', replace(radar, enemy=(), explored=np.ones_like(radar.explored)))
    ctx.radar_home = radar.own[0][1]
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    world.set('credits', 700)
    state, plan = army.miners, army.book['resources']
    state.grouped = True
    state.home_ore_thin_since = time[0] - plan['home_thin_seconds']
    army.intent.resources = 'survey_gems'
    monkeypatch.setattr(miners, 'gem_field', lambda *args: None if ctx.camera == 'home' else ((400, 350), 12))

    class SurveyCamera(CameraGame):
        def __init__(self, world):
            super().__init__(world)
            self.commands = []

        def act(self, name, lane, steps, commands=1):
            self.commands.append(commands)
            super().act(name, lane, steps, commands)

    executor = runner.executor = SurveyCamera(world)
    assert army.step(world.snapshot())[0] == 'army:resource-survey'
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert ctx.camera == 'home' and state.field is not None
    assert executor.commands and not any(executor.commands)
    assert world.events[-1]['kind'] == 'resource_survey' and world.events[-1]['found']
    assert not miners.survey_due(army, time[0])


def test_watch_camera_records_screen_observed_field_on_radar(monkeypatch):
    world, army, reflex, time, state, plan = ready()
    radar = online_radar()
    world.set('radar', replace(radar, explored=np.ones_like(radar.explored)))
    army.ctx.radar_home = radar.own[0][1]
    army.ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    monkeypatch.setattr(miners, 'gem_field', lambda *args: ((400, 350), 12))
    miners.observe_field(army.ctx, state, plan, time[0] + 3)
    assert state.field is not None and state.field_chips == 12
    assert state.field_at == time[0] + 3
    assert world.events[-1]['kind'] == 'resource_field'


def test_systemone_choice_dispatches_move_then_fresh_gem_order(monkeypatch):
    world, army, reflex, time, state, plan = ready()
    army.intent.resources = 'harvest_gems'
    monkeypatch.setattr(miners, 'selected', lambda ctx: [vehicle()])
    assert miners.resource_due(army, time[0])
    actions = drive(miners.resource_order(army.ctx, state, plan, lambda: time[0]), army.ctx, time)
    assert state.resource_phase == 'travel'
    assert any(s.kind == 'click' and s.point == (920, 80) for s in actions)

    time[0] += plan['travel_wait']
    assert miners.resource_due(army, time[0])
    monkeypatch.setattr(miners, 'gem_field', lambda *args: ((430, 320), 8))
    actions = drive(miners.resource_order(army.ctx, state, plan, lambda: time[0]), army.ctx, time)
    assert state.resource_phase == 'ordered'
    assert any(s.kind == 'click' and s.point == (430, 320) for s in actions)
    assert world.events[-1]['status'] == 'gem harvest order sent'


@pytest.mark.parametrize('failure', ['missing', 'unhealthy', 'no_gems'])
def test_missing_miner_or_unverified_crystals_cannot_issue_harvest(monkeypatch, failure):
    world, army, reflex, time, state, plan = ready()
    state.resource_phase, state.resource_at = 'travel', time[0] - plan['travel_timeout']
    selection = [] if failure == 'missing' else [vehicle(0.2 if failure == 'unhealthy' else 1)]
    monkeypatch.setattr(miners, 'selected', lambda ctx: selection)
    monkeypatch.setattr(miners, 'gem_field', lambda *args: None)
    actions = drive(miners.resource_order(army.ctx, state, plan, lambda: time[0]), army.ctx, time)
    assert state.resource_phase == 'idle'
    assert not any(s.kind == 'click' and s.button == 'left' for s in actions)
    if failure == 'missing':
        assert not state.grouped


def test_cash_rise_is_logged_without_claiming_verified_delivery():
    world, army, reflex, time, state, plan = ready()
    state.resource_phase, state.resource_at = 'ordered', time[0]
    state.resource_credits = 700
    world.set('credits', 900)
    miners.monitor_income(army.ctx, state, plan, time[0] + 1)
    assert state.resource_phase == 'idle'
    assert 'source unverified' in world.events[-1]['status']


def stalled_income():
    world, army, reflex, time, state, plan = ready()
    world.set('credits', 0)
    state.home_ore_seen, state.home_ore_thin_since = True, None
    state.home_ore_target = (100, 40)
    lane = world.lanes['buildings']
    lane.status, lane.item, lane.progress = 'building', 'ore_refinery', 0.685
    miners.monitor_income(army.ctx, state, plan, time[0])
    time[0] += 91
    state.home_ore_at = time[0]
    miners.monitor_income(army.ctx, state, plan, time[0])
    return world, army, reflex, time, state, plan


def test_stalled_income_offers_verified_home_ore_despite_plentiful_visible_chips():
    world, army, reflex, time, state, plan = stalled_income()
    assert miners.income_stalled(state, plan, time[0])
    options = reflex.feasible_resources(time[0])
    assert 'harvest_home_ore' in options
    assert reflex.book.rule('resources', options, reflex.resource_facts(time[0])[1]) == 'harvest_home_ore'
    army.intent.resources = 'harvest_home_ore'
    assert miners.resource_due(army, time[0])


@pytest.mark.parametrize('activity', ['cash', 'progress', 'completed'])
def test_cash_or_production_progress_clears_income_stall(activity):
    world, army, reflex, time, state, plan = stalled_income()
    if activity == 'cash':
        world.set('credits', 1)
    elif activity == 'progress':
        world.lanes['buildings'].progress = 0.73
    else:
        world.lanes['buildings'].status = 'ready'
    miners.monitor_income(army.ctx, state, plan, time[0] + 1)
    assert not miners.income_stalled(state, plan, time[0] + 1)
    assert 'harvest_home_ore' not in reflex.feasible_resources(time[0] + 1)


def test_small_progress_noise_does_not_hide_a_stall_but_cumulative_progress_does():
    world, army, reflex, time, state, plan = stalled_income()
    lane = world.lanes['buildings']
    for delta in (0.01, 0.02):
        lane.progress = 0.685 + delta
        miners.monitor_income(army.ctx, state, plan, time[0] + 1)
        assert miners.income_stalled(state, plan, time[0] + 1)
    lane.progress = 0.725
    miners.monitor_income(army.ctx, state, plan, time[0] + 2)
    assert not miners.income_stalled(state, plan, time[0] + 2)


def test_next_copy_of_the_same_queued_unit_gets_a_new_progress_baseline():
    world, army, reflex, time, state, plan = stalled_income()
    lane = world.lanes['buildings']
    lane.progress = 0.05
    miners.monitor_income(army.ctx, state, plan, time[0] + 1)
    assert miners.income_stalled(state, plan, time[0] + 1)
    lane.progress = 0.1
    miners.monitor_income(army.ctx, state, plan, time[0] + 2)
    assert not miners.income_stalled(state, plan, time[0] + 2)


@pytest.mark.parametrize('missing', ['ore', 'fresh_view', 'miner', 'quiet', 'campaign'])
def test_income_recovery_needs_fresh_ore_a_saved_miner_and_a_quiet_base(missing):
    world, army, reflex, time, state, plan = stalled_income()
    if missing == 'ore':
        state.home_ore_target = None
    elif missing == 'fresh_view':
        state.home_ore_at -= plan['home_fresh_within'] + 1
    elif missing == 'miner':
        state.grouped = False
    elif missing == 'quiet':
        army.threat_at = time[0]
    else:
        army.campaign.active = True
    assert 'harvest_home_ore' not in reflex.feasible_resources(time[0])
    army.intent.resources = 'harvest_home_ore'
    assert not miners.resource_due(army, time[0])


@pytest.mark.parametrize('missing', [None, 'miner', 'health', 'ore'])
def test_home_ore_order_rechecks_miner_and_clicks_only_fresh_crystals(monkeypatch, missing):
    world, army, reflex, time, state, plan = stalled_income()
    selection = [] if missing == 'miner' else [vehicle(0.2 if missing == 'health' else 1)]
    monkeypatch.setattr(miners, 'selected', lambda ctx: selection)
    monkeypatch.setattr(miners, 'ore_target', lambda *args: None if missing == 'ore' else (150, 50))
    actions = drive(miners.resource_order(army.ctx, state, plan, lambda: time[0], resource='ore'),
                    army.ctx, time)
    orders = [s for s in actions if s.kind == 'click' and s.button == 'left']
    assert len(orders) == (0 if missing else 1)
    if not missing:
        assert orders[0].point == (150, 50) and state.resource_phase == 'ordered'
    assert army.ctx.camera == 'home'
