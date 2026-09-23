"""Optional Soviet tech requires economy and recent battlefield evidence."""

import copy

import pytest

from addons.chronodivide import playbook
from addons.chronodivide.intel import Intel
from addons.chronodivide.tests.test_playbook import game


def test_soviet_faction_knowledge_includes_shared_troopers_and_optional_desolators():
    book = playbook.load()
    knowledge = playbook.knowledge(book)
    assert book.data['faction'] == knowledge['faction'] == 'soviet'
    assert 'country' not in book.data
    assert 'tesla_tank' not in knowledge['items']
    assert 'tesla_trooper' in knowledge['items']
    desolator = knowledge['items']['desolator']
    assert desolator['lane'] == 'infantry' and desolator['cost'] == 600
    assert set(desolator['requires']) == {'barracks', 'radar_tower'}


def test_desolator_production_requires_observed_sidebar_availability():
    world, army, reflex, time = advantaged()
    assert reflex.next_item('infantry', ['tesla_trooper'], time[0]) is None
    assert reflex.next_item('infantry', ['desolator'], time[0]) == 'desolator'


def advantaged(book=None):
    world, army, reflex, time = game(book)
    world.structures.update(construction_yard=1, tesla_reactor=2, ore_refinery=2,
                            barracks=1, war_factory=2, radar_tower=1)
    world.units_built.update(war_miner=2, rhino_tank=12, flak_track=6,
                            attack_dog=2, conscript=6)
    world.set('credits', 6000)
    army.census_tanks, army.census_at, army.census_healthy_vehicles = 12, time[0], 10
    army.intel = Intel(lambda: time[0], book)
    army.intel.enemy_base = (970, 108)
    army.intel.record_look('enemy_base', (970, 108),
                         dict(enemy_objects=2, enemy_pixels=200, own_objects=0, shroud=0.1), time[0])
    return world, army, reflex, time


@pytest.mark.parametrize('missing', ['cash', 'refineries', 'miners', 'quiet', 'army',
                                   'health', 'census', 'intel', 'enemy', 'visibility', 'strength'])
def test_each_missing_economy_or_advantage_fact_blocks_all_high_tech(missing):
    world, army, reflex, time = advantaged()
    plan = reflex.book['high_tech']
    if missing == 'cash':
        world.set('credits', plan['min_credits'] - 1)
    elif missing == 'refineries':
        world.structures['ore_refinery'] = plan['min_refineries'] - 1
    elif missing == 'miners':
        world.units_built['war_miner'] = 0
    elif missing == 'quiet':
        army.threat_at = time[0]
    elif missing == 'army':
        army.census_tanks = plan['min_army'] - 1
    elif missing == 'health':
        army.census_healthy_vehicles = plan['min_healthy_vehicles'] - 1
    elif missing == 'census':
        army.census_at = time[0] - plan['fresh_within'] - 1
    elif missing == 'intel':
        army.intel.looks['enemy_base']['at'] = time[0] - plan['fresh_within'] - 1
    elif missing == 'enemy':
        army.intel.looks['enemy_base']['enemy_objects'] = 0
    elif missing == 'visibility':
        army.intel.looks['enemy_base']['shroud'] = 0.9
    elif missing == 'strength':
        army.intel.strength = 'strong'
    army.intent.set('objective', 'tech_up', 'old answer')
    assert not reflex.high_tech_ready(time[0])
    assert 'tech_up' not in reflex.feasible_objectives()
    order = reflex.book['buildings']['order']
    lab = next(e for e in order if e['item'] == 'battle_lab')
    power = next(e for e in order if e.get('with') == 'battle_lab')
    assert not reflex.entry_due(lab, time[0], order)
    assert not reflex.entry_due(power, time[0], order)
    assert reflex.next_item('buildings', ['tesla_reactor', 'battle_lab'], time[0]) != 'battle_lab'
    vehicles, _ = reflex.unit_options('vehicles', ['rhino_tank', 'apocalypse_tank'], 6000, time[0])
    assert 'apocalypse_tank' not in vehicles
    assert reflex.next_item('infantry', ['desolator', 'conscript'], time[0]) is None


def test_economy_and_fresh_advantage_allow_tech_and_advanced_units():
    world, army, reflex, time = advantaged()
    assert reflex.high_tech_ready(time[0])
    assert 'tech_up' in reflex.feasible_objectives()
    army.intent.set('objective', 'tech_up', 'systemone')
    assert reflex.next_item('buildings', ['tesla_reactor', 'battle_lab'], time[0]) == 'battle_lab'
    assert reflex.next_item('vehicles', ['rhino_tank', 'apocalypse_tank'], time[0]) == 'apocalypse_tank'
    assert reflex.next_item('infantry', ['desolator', 'conscript'], time[0]) == 'desolator'
    army.threat_at = time[0]
    assert reflex.next_item('vehicles', ['rhino_tank', 'apocalypse_tank'], time[0]) == 'rhino_tank'


def test_tech_thresholds_are_read_from_selected_json_profile():
    data = copy.deepcopy(playbook.load().data)
    data['playbook']['high_tech'].update(min_credits=7000, min_army=14)
    world, army, reflex, time = advantaged(playbook.Playbook(data))
    assert not reflex.high_tech_ready(time[0])
    world.set('credits', 7000)
    assert not reflex.high_tech_ready(time[0])
    army.census_tanks = 14
    assert reflex.high_tech_ready(time[0])
