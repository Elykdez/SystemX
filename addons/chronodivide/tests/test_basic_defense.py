"""Basic turrets supplement mobile production without taking its next vehicle budget."""

from types import SimpleNamespace

import pytest

from addons.chronodivide import skills
from addons.chronodivide.layout import Layout
from addons.chronodivide.reflex import Reflex
from addons.chronodivide.world import Slot, World

NOW = 1000.0


def setup(credits=1400, turrets=0, threat_age=float('inf')):
    world = World(clock=lambda: NOW)
    world.set('credits', credits)
    world.set('power', {'level': 'ok'})
    world.structures.update(construction_yard=1, barracks=1, tesla_reactor=2,
                            ore_refinery=2, war_factory=1, sentry_gun=turrets)
    world.units_built.update(rhino_tank=4, war_miner=2)
    reflex = Reflex(skills.Context(Layout.for_canvas(1024, 768), world), clock=lambda: NOW)
    reflex.army = SimpleNamespace(threat_age=lambda now: threat_age)
    return world, reflex


def choice(reflex):
    return reflex.next_item('defenses', ['sentry_gun', 'tesla_coil'], NOW)


def test_first_cheap_turret_does_not_wait_for_a_raid_and_preserves_one_rhino():
    world, reflex = setup()
    world.units_built['rhino_tank'] = 2
    assert choice(reflex) == 'sentry_gun'
    assert world.spendable() - world.item('sentry_gun')['cost'] == world.item('rhino_tank')['cost']
    options, needs = reflex.unit_options('defenses', ['sentry_gun', 'tesla_coil'], 1400, NOW)
    assert options == ['sentry_gun'] and 'basic base cover' in needs[0]
    assert 'basic base cover' in reflex.book.option('production', 'sentry_gun')['words']


@pytest.mark.parametrize('missing', ['war_factory', 'ore_refinery', 'two_rhinos'])
def test_opening_and_lost_economy_do_not_spend_on_turrets(missing):
    world, reflex = setup(10000, threat_age=0)
    if missing == 'two_rhinos':
        world.units_built['rhino_tank'] = 1
        world.units_built['apocalypse_tank'] = 4
    else:
        world.structures[missing] = 0
    assert choice(reflex) is None


@pytest.mark.parametrize('credits', [None, 0, 500, 900, 1399])
def test_poor_or_unknown_balance_never_consumes_mobile_reserve(credits):
    _, reflex = setup(credits, threat_age=0)
    assert choice(reflex) is None


def test_only_one_turret_is_offered_even_during_a_raid():
    for turrets, threat, expected in [(0, 61, 'sentry_gun'), (1, 60, None),
                                       (2, 0, None), (4, 0, None)]:
        _, reflex = setup(10000, turrets, threat)
        assert choice(reflex) == expected
    world, reflex = setup(10000, 1, 0)
    world.structures['tesla_coil'] = 1
    assert choice(reflex) is None


def test_second_refinery_savings_and_needed_miners_have_priority():
    world, reflex = setup(1900)
    world.units_built['rhino_tank'] = reflex.book['units']['second_refinery']['after_tanks']
    world.structures['ore_refinery'] = 1
    assert choice(reflex) is None
    world.set('credits', 3400)
    assert choice(reflex) == 'sentry_gun'  # 2000 refinery + 900 Rhino remain after 500 gun
    world.structures['ore_refinery'] = 2
    world.units_built['war_miner'] = 0
    world.set('credits', 1899)
    assert choice(reflex) is None
    world.set('credits', 1900)
    assert choice(reflex) == 'sentry_gun'  # the needed 1400 miner can still be bought


def test_second_refinery_waits_for_a_tank_reserve_during_a_raid():
    world, reflex = setup(2500, threat_age=0)
    world.structures['ore_refinery'] = 1
    world.structures.update(tesla_reactor=2, radar_tower=1)
    world.units_built['rhino_tank'] = reflex.book['units']['second_refinery']['after_tanks']
    available = ['ore_refinery', 'tesla_reactor', 'radar_tower']
    assert reflex.next_building(available, NOW) is None
    world.set('credits', 2900)
    assert reflex.next_building(available, NOW) == 'ore_refinery'
    reflex.army = SimpleNamespace(threat_age=lambda now: 31)
    world.set('credits', 2000)
    assert reflex.next_building(available, NOW) == 'ore_refinery'


def test_recent_raid_keeps_tank_reserve_instead_of_saving_for_miners():
    world, reflex = setup(1400, 0, 0)
    world.units_built['war_miner'] = 0
    assert choice(reflex) == 'sentry_gun'


def test_concurrent_vehicle_reservations_are_subtracted_before_turret_budget():
    world, reflex = setup(2299)
    tank = world.ledger.add('queue', 'vehicles', 'rhino_tank', 900)
    assert world.spendable() == 1399 and choice(reflex) is None
    world.set('credits', 2300)
    assert choice(reflex) == 'sentry_gun'
    defense = world.ledger.add('queue', 'defenses', 'sentry_gun', 500)
    world.set('credits', 10000)
    assert choice(reflex) is None  # no duplicate while its first click awaits confirmation
    world.ledger.close(defense, 'cancelled')
    world.ledger.close(tank, 'cancelled')
    assert choice(reflex) == 'sentry_gun'


@pytest.mark.parametrize('status', ['building', 'ready', 'hold'])
def test_active_turret_production_must_finish_before_another_is_considered(status):
    world, reflex = setup(10000, 0, 0)
    world.update_sidebar('defenses', [Slot(0, 'sentry_gun', status)])
    assert choice(reflex) is None


def test_first_turret_makes_defense_tab_reachable_before_a_raid():
    world, reflex = setup()
    for lane, item in [('buildings', 'tesla_reactor'), ('vehicles', 'rhino_tank'), ('infantry', 'conscript')]:
        world.update_sidebar(lane, [Slot(0, item, 'building')])
    assert reflex.plan(world.snapshot(), NOW)[0] == 'look:defenses'
    world.update_sidebar('defenses', [Slot(0, 'sentry_gun', 'available')])
    assert reflex.plan(world.snapshot(), NOW)[0] == 'queue:sentry_gun'


def test_turret_choice_needs_no_model_roundtrip():
    _, reflex = setup()
    reflex.broker = SimpleNamespace(submit=lambda _: pytest.fail('single choice queried the model'))
    assert choice(reflex) == 'sentry_gun'


def test_low_power_keeps_reactor_recovery_ahead_of_extra_cover():
    world, reflex = setup(10000)
    world.set('power', {'level': 'low'})
    assert choice(reflex) is None
