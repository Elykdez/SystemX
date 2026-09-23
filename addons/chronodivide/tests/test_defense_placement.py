"""Defensive structures probe the known approach first, still requiring a valid footprint."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from addons.chronodivide import playbook, skills
from addons.chronodivide.layout import Layout
from addons.chronodivide.tasks import Act, Look, Poll, Wait
from addons.chronodivide.world import Slot, World

EXTENT = (368, 320, 488, 400)


def place(monkeypatch, item, first_blocked=False, radar=True, front=None, knowledge=None, west=False,
          red_block=False, factory_exit=None):
    world = World(knowledge=knowledge, clock=lambda: 100.0)
    ctx = skills.Context(Layout.for_canvas(1024, 768), world)
    world.update_sidebar(ctx.lane(item), [Slot(0, item, 'ready')])
    ctx.last_canvas = np.full((768, 1024, 3), 100, np.uint8)
    if item == 'war_factory':
        ctx.last_canvas[:] = (50, 140, 90)
    ctx.factory_exit = factory_exit
    if radar:
        world.set('radar', SimpleNamespace(online=True, center=(900 if west else 970, 100), summary=lambda: {}))
        ctx.radar_home = (970 if west else 900, 100)
    original = ctx.placement_candidates(EXTENT)
    monkeypatch.setattr(skills, 'center_base', lambda ctx: iter(()))
    monkeypatch.setattr(skills, 'base_extent', lambda *args: EXTENT)
    probes, clicks = [], []

    def ghost(canvas, reference, point):
        probes.append(point)
        return SimpleNamespace(seen=True, valid=not first_blocked or len(probes) > 1,
                               red_center=(20, 10) if red_block else None,
                               green_center=(0, 10) if red_block else None)

    monkeypatch.setattr(skills, 'read_ghost', ghost)
    task = skills.place_building(ctx, item, front=front)
    response = None
    while True:
        try:
            step = task.send(response)
        except StopIteration:
            break
        response = None
        if isinstance(step, Look):
            response = step.fn(world.snapshot(), None)
        elif isinstance(step, Poll):
            response = step.reason == 'placement'
        elif isinstance(step, Wait):
            response = step.until(world.snapshot(), None)
            assert response
        elif isinstance(step, Act):
            clicks.extend(s.point for s in step.steps if s.kind == 'click')
        else:
            pytest.fail(f'Unexpected placement step: {step}')
    return ctx, original, probes, clicks


@pytest.mark.parametrize('item', ['sentry_gun', 'tesla_coil', 'flak_cannon'])
def test_defense_first_probe_is_nearest_known_front(monkeypatch, item):
    ctx, original, probes, clicks = place(monkeypatch, item)
    front = ctx.front_point()
    nearest = min(original, key=lambda p: (p[0] - front[0]) ** 2 + (p[1] - front[1]) ** 2)
    assert probes == clicks == [nearest]
    assert nearest[0] > ctx.home[0] and nearest != original[0]
    assert ctx.world.structures[item] == 1


def test_other_construction_keeps_existing_candidate_order(monkeypatch):
    _, original, probes, clicks = place(monkeypatch, 'ore_refinery')
    assert probes == clicks == [original[0]]


def test_factory_requires_open_ground_beyond_its_door():
    canvas = np.full((768, 1024, 3), (50, 140, 90), np.uint8)
    assert skills.factory_access_open(canvas, (238, 345))
    canvas[380:450, 250:350] = (100, 100, 100)  # structures box in the exit
    assert not skills.factory_access_open(canvas, (238, 345))
    assert skills.factory_access_open(canvas, (500, 345))


def test_placed_factory_reserves_its_exit(monkeypatch):
    ctx, _, probes, _ = place(monkeypatch, 'war_factory')
    assert ctx.factory_exit == (probes[-1][0] + 60, probes[-1][1] + 60)


@pytest.mark.parametrize('faction,item', [('allied', 'power_plant'), ('soviet', 'tesla_reactor'),
                                         ('allied', 'ore_refinery'), ('soviet', 'barracks')])
def test_new_buildings_preserve_the_factory_access(monkeypatch, faction, item):
    exit_point = (298, 435)
    book = playbook.load(Path(__file__).parents[1] / 'profiles' / f'startlux-ra2-{faction}.json')
    _, original, probes, clicks = place(monkeypatch, item, factory_exit=exit_point,
                                       knowledge=playbook.knowledge(book))
    assert original[0] == exit_point and original[0] not in probes
    assert clicks == [probes[-1]]
    assert all(abs(x - exit_point[0]) + 2 * abs(y - exit_point[1]) >= 180 for x, y in probes)


@pytest.mark.parametrize('radar', [False, True])
@pytest.mark.parametrize('blocked', [False, True])
def test_allied_lab_probes_behind_the_western_approach_and_requires_a_green_footprint(monkeypatch, radar, blocked):
    book = playbook.load(Path(__file__).parents[1] / 'profiles' / 'startlux-ra2-allied.json')
    ctx, original, probes, clicks = place(monkeypatch, 'battle_lab', first_blocked=blocked,
                                          radar=radar, knowledge=playbook.knowledge(book), west=True)
    assert len(probes) == (2 if blocked else 1)
    assert clicks == [probes[-1]]
    if radar:
        assert all(point[0] >= ctx.home[0] for point in probes)
        assert original[0][0] < ctx.home[0]  # The previous first site faced the incoming army.
    else:
        assert probes == original[:len(probes)]
    assert ctx.world.structures['battle_lab'] == 1


def test_blocked_front_footprint_is_never_clicked(monkeypatch):
    ctx, original, probes, clicks = place(monkeypatch, 'sentry_gun', first_blocked=True)
    front = ctx.front_point()
    ordered = sorted(original, key=lambda p: (p[0] - front[0]) ** 2 + (p[1] - front[1]) ** 2)
    assert probes == ordered[:2]
    assert clicks == [ordered[1]] and ordered[0] not in clicks


def test_blocked_lab_detour_does_not_jump_ahead_of_remaining_rear_sites(monkeypatch):
    book = playbook.load(Path(__file__).parents[1] / 'profiles' / 'startlux-ra2-allied.json')
    ctx, original, probes, clicks = place(monkeypatch, 'battle_lab', first_blocked=True,
                                          knowledge=playbook.knowledge(book), west=True, red_block=True)
    detour = (probes[0][0] - skills.CELL[0], probes[0][1])
    assert detour[0] < ctx.home[0]
    assert len(probes) == 2 and clicks == [probes[-1]]
    assert all(point[0] >= ctx.home[0] for point in probes)


def test_without_radar_defense_uses_existing_front_fallback(monkeypatch):
    ctx, original, probes, clicks = place(monkeypatch, 'sentry_gun', radar=False)
    front = ctx.front_point()
    nearest = min(original, key=lambda p: (p[0] - front[0]) ** 2 + (p[1] - front[1]) ** 2)
    assert probes == clicks == [nearest]


def test_observed_raid_overrides_fallback_but_still_requires_green_footprint(monkeypatch):
    front = (740, 470)
    ctx, original, probes, clicks = place(monkeypatch, 'sentry_gun', first_blocked=True,
                                          radar=False, front=front)
    ordered = sorted(original, key=lambda p: (p[0] - front[0]) ** 2 + (p[1] - front[1]) ** 2)
    assert probes == ordered[:2] and clicks == [ordered[1]]
    assert probes[0][0] > ctx.home[0]


@pytest.mark.parametrize('condition', ['fresh', 'stale', 'away', 'menu', 'missing', 'other_building'])
def test_reflex_uses_only_fresh_home_raid_for_defensive_placement(monkeypatch, condition):
    from addons.chronodivide.tests.test_policy import setup, show
    world, ctx, runner, reflex, executor, time = setup()
    world.set('mode', 'hud')
    item = 'ore_refinery' if condition == 'other_building' else 'sentry_gun'
    show(world, ctx.lane(item), (item, 'ready', 1.0))
    reflex.army.seen = ((740, 470), 0)
    reflex.army.last_check = time[0]
    if condition == 'stale':
        reflex.army.last_check -= reflex.army.every + 0.01
    elif condition == 'away':
        ctx.camera = 'away'
    elif condition == 'menu':
        world.set('mode', 'menu')
    elif condition == 'missing':
        reflex.army.seen = None
    received = []

    def placement(ctx, item, ring, front=None):
        received.append(front)
        return iter(())

    monkeypatch.setattr(skills, 'place_building', placement)
    plan = reflex.plan(world.snapshot(), time[0])
    plan[1]()
    offset = reflex.book['units']['defenses']['front_without_radar']
    fallback = (ctx.home[0] + offset[0], ctx.home[1] + offset[1])
    expected = (740, 470) if condition == 'fresh' else None if condition == 'other_building' else fallback
    assert received == [expected]
