"""Basic anti-air infantry uses its own configured gates, below optional high tech."""

import pytest

from addons.chronodivide import playbook
from addons.chronodivide.tests.test_policy import setup
from addons.chronodivide.tests.test_strategy import Broker


def ready():
    world, ctx, runner, reflex, executor, time = setup()
    world.units_built['attack_dog'] = reflex.book['units']['scouts']['count']
    world.set('credits', 1200)
    return world, reflex, time


@pytest.mark.parametrize('tanks,trained,credits,offered,expected', [
    (1, 0, 1200, True, False),
    (2, 0, 1199, True, False),
    (2, 0, 1200, True, True),
    (2, 1, 1200, True, True),
    (2, 2, 1200, True, False),
    (20, 11, 1200, True, True),
    (20, 12, 1200, True, False),
    (2, 0, 1200, False, False),
])
def test_basic_anti_air_respects_tank_cash_ratio_cap_and_sidebar_boundaries(tanks, trained, credits, offered, expected):
    world, reflex, time = ready()
    world.units_built.update(rhino_tank=tanks, flak_trooper=trained)
    world.set('credits', credits)
    available = ['conscript', 'flak_trooper'] if offered else ['conscript']
    assert reflex.next_item('infantry', available, time[0]) == ('flak_trooper' if expected else 'conscript')


def test_basic_anti_air_waits_for_dog_quota_and_grouping():
    world, reflex, time = ready()
    world.units_built['rhino_tank'] = 2
    world.units_built['attack_dog'] -= 1
    available = ['attack_dog', 'flak_trooper', 'conscript']
    assert reflex.next_item('infantry', available, time[0]) == 'attack_dog'
    world.units_built['attack_dog'] += 1
    reflex.army.dogs_at = time[0]
    assert reflex.next_item('infantry', available, time[0]) is None
    reflex.army.scouts_ready = True
    assert reflex.next_item('infantry', available, time[0]) == 'flak_trooper'


def test_basic_anti_air_continues_after_conscripts_without_high_tech_readiness():
    world, reflex, time = ready()
    world.set('credits', 6000)
    world.units_built.update(rhino_tank=2, conscript=reflex.book['units']['conscripts']['early'])
    assert not reflex.high_tech_ready(time[0])
    assert reflex.next_item('infantry', ['flak_trooper', 'conscript', 'tesla_trooper'], time[0]) == 'flak_trooper'


def test_profile_changes_basic_anti_air_gates_and_ratio():
    world, reflex, time = ready()
    reflex.book = playbook.load()
    reflex.book['units']['anti_air_infantry'].update(after_tanks=3, one_per_tanks=2, max=3, credits=900)
    world.set('credits', 899)
    world.units_built.update(rhino_tank=3, flak_trooper=1)
    available = ['flak_trooper']
    assert reflex.next_item('infantry', available, time[0]) is None
    world.set('credits', 900)
    assert reflex.next_item('infantry', available, time[0]) == 'flak_trooper'
    world.units_built['flak_trooper'] = 2
    assert reflex.next_item('infantry', available, time[0]) is None
    world.units_built.update(rhino_tank=7, flak_trooper=3)
    assert reflex.next_item('infantry', available, time[0]) is None


def test_production_question_names_infantry_support_and_defaults_to_it():
    world, reflex, time = ready()
    world.units_built['rhino_tank'] = 2
    reflex.broker = Broker(time)
    assert reflex.next_item('infantry', ['flak_trooper', 'conscript'], time[0]) is None
    decision = reflex.broker.submitted[-1]
    criteria = decision.questions['production']['criteria']
    assert list(criteria) == ['flak_trooper', 'conscript']
    assert 'anti-air infantry support' in criteria['flak_trooper']
    assert decision.default['production']['choice'] == 'flak_trooper'
    assert any('Flak Troopers' in need for need in decision.state['needs'])
    assert all('Flak Tracks' not in need for need in decision.state['needs'])
