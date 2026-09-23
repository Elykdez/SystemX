"""Recent raids keep optional tech from consuming the next tank's money."""

import pytest

from addons.chronodivide import skills
from addons.chronodivide.tests.test_policy import setup, show


@pytest.mark.parametrize('reactors,item', [(1, 'tesla_reactor'), (2, 'radar_tower')])
def test_optional_radar_and_its_reactor_preserve_tank_money(reactors, item):
    world, ctx, runner, reflex, executor, time = setup()
    world.structures.update(construction_yard=1, tesla_reactor=reactors, ore_refinery=2,
                            barracks=1, war_factory=1)
    world.units_built['rhino_tank'] = 4
    world.set('power', {'level': 'ok'})
    reflex.army.threat_at = time[0]
    world.set('credits', 1050)
    available = ['tesla_reactor', 'radar_tower']
    assert reflex.next_item('buildings', available, time[0]) is None
    threshold = world.item(item)['cost'] + world.item(reflex.book['units']['tank'])['cost']
    world.set('credits', threshold - 1)
    assert reflex.next_item('buildings', available, time[0]) is None
    world.set('credits', threshold)
    assert reflex.next_item('buildings', available, time[0]) == item
    world.set('credits', 1050)
    assert reflex.next_item('buildings', available, time[0] + 31) == item
    # Repairing actual low power is independent of optional technology preparation.
    world.set('power', {'level': 'low'})
    assert reflex.next_item('buildings', available, time[0]) == 'tesla_reactor'


def test_tech_up_objective_does_not_bypass_mobile_reserve():
    world, ctx, runner, reflex, executor, time = setup()
    world.units_built['rhino_tank'] = 10
    reflex.intent.objective = 'tech_up'
    reflex.army.threat_at = time[0]
    world.set('credits', 1050)
    assert reflex.next_item('buildings', ['radar_tower'], time[0]) is None


def test_rich_vehicle_batches_do_not_exceed_needed_support_units():
    world, ctx, runner, reflex, executor, time = setup()
    world.structures.update(ore_refinery=2, war_factory=1)
    world.units_built.update(rhino_tank=2, war_miner=1)
    world.set('credits', 10000)
    assert reflex.queue_count('rhino_tank') == 2
    assert reflex.queue_count('flak_track') == 1  # one Flak supports the first two Rhinos
    assert reflex.queue_count('war_miner') == 1  # two refinery miners plus one purchased miner
    world.units_built.update(rhino_tank=6, flak_track=1)
    assert reflex.queue_count('flak_track') == 1  # two tracks cover six Rhinos at one per four
    world.units_built['rhino_tank'] = 10
    assert reflex.queue_count('flak_track') == 2  # both remaining support units are needed


def test_flak_infantry_preserves_money_for_the_next_rhino():
    world, ctx, runner, reflex, executor, time = setup()
    world.units_built.update(attack_dog=2, conscript=2, rhino_tank=4)
    available = ['flak_trooper', 'conscript', 'attack_dog']
    reserve = world.item('flak_trooper')['cost'] + world.item('rhino_tank')['cost']
    assert 'flak_trooper' not in reflex.unit_options('infantry', available, reserve - 1, time[0])[0]
    assert 'flak_trooper' in reflex.unit_options('infantry', available, reserve, time[0])[0]


@pytest.mark.parametrize('count,credits,allowed', [(1, 853, False), (1, 900, True),
                                                  (2, 1799, False), (2, 1800, True)])
def test_vehicle_queue_rechecks_money_after_switching_tabs(count, credits, allowed):
    world, ctx, runner, reflex, executor, time = setup()
    world.set('credits', 3000)
    show(world, 'buildings', ('tesla_reactor', 'building', 0.1))
    runner.start('queue:rhino_tank', 'sidebar', skills.queue_item(ctx, 'rhino_tank', count))
    runner.tick(world.snapshot(), None)
    assert executor.acts[-1][1][0][2] == ('r',)
    # Construction spent credits while the vehicle tab was opening.
    world.set('credits', credits)
    show(world, 'vehicles', ('rhino_tank', 'available', None))
    runner.tick(world.snapshot(), None)
    clicks = [step for _, steps in executor.acts for step in steps if step[0] == 'click']
    assert len(clicks) == (count if allowed else 0)
    if not allowed:
        assert runner.finished[-1].status == 'failed'
        assert not world.ledger.items and not world.units_built


@pytest.mark.parametrize('blocked', ['sidebar', 'cooldown', 'recommend'])
def test_production_choice_survives_until_queue_task_starts(monkeypatch, blocked):
    from addons.chronodivide.tests.test_strategy import Broker

    world, ctx, runner, reflex, executor, time = setup()
    world.structures.update(war_factory=1, ore_refinery=2)
    world.units_built.update(rhino_tank=9, flak_track=2, war_miner=2)
    world.set('credits', 1875)
    reflex.army.threat_at = time[0]
    reflex.broker = Broker(time)
    monkeypatch.setattr(reflex, 'ask_strategy', lambda now: None)
    show(world, 'vehicles', ('rhino_tank', 'available', None), ('flak_track', 'available', None))
    reflex.choices['vehicles'] = ('flak_track', 'systemone')
    if blocked == 'sidebar':
        runner.start('look:vehicles', 'sidebar', (step for step in ()))
    elif blocked == 'cooldown':
        reflex.cooldown['queue:flak_track'] = time[0] + 1

    assert reflex.step(world.snapshot(), act=blocked != 'recommend')[0] == 'queue:flak_track'
    assert reflex.choices['vehicles'] == ('flak_track', 'systemone')
    assert not reflex.broker.submitted
    runner.tick(world.snapshot(), None)
    time[0] += 1.1
    show(world, 'vehicles', ('rhino_tank', 'available', None), ('flak_track', 'available', None))

    assert reflex.step(world.snapshot())[0] == 'queue:flak_track'
    assert runner.tasks['sidebar'].name == 'queue:flak_track'
    assert 'vehicles' not in reflex.choices
    assert not reflex.broker.submitted
    runner.tick(world.snapshot(), None)
    assert executor.acts[-1][1][0][1] == ctx.slot_point(1)
