"""The System One profile: questions, fact words, rule defaults and playbook numbers come from JSON."""

import copy

import pytest

from addons.chronodivide import playbook
from addons.chronodivide.army import Army
from addons.chronodivide.intent import StandingIntent
from addons.chronodivide.layout import Layout
from addons.chronodivide.reflex import Reflex
from addons.chronodivide.skills import Context
from addons.chronodivide.world import World


def game(book=None):
    time = [500.0]
    world = World(clock=lambda: time[0])
    ctx = Context(Layout.for_canvas(1024, 768), world)
    intent = StandingIntent()
    army = Army(ctx, None, intent, clock=lambda: time[0], book=book)
    reflex = Reflex(ctx, None, None, intent, clock=lambda: time[0], army=army, book=book)
    return world, army, reflex, time


def test_the_shipped_profile_words_every_question_and_its_options():
    book = playbook.load()
    for qid in ('objective', 'posture', 'attack_size', 'intel', 'production'):
        spec = book.questions[qid]
        assert spec['instructions'] and spec['facts'] and len(spec['options']) >= 2
        question = book.question(qid, book.options(qid)[:2])[qid]
        assert question['type'] == 'choice' and all(question['criteria'].values())
    assert [book.option('attack_size', key)['tanks'] for key in book.options('attack_size')] == [4, 5, 12, 14, 16]


def test_soviet_assault_does_not_recount_home_reinforcements_before_combat_can_advance():
    _, army, _, clock = game()
    army.census_at = clock[0] - 30
    assert army._census_due(clock[0])
    army.campaign.active = True
    assert not army._census_due(clock[0])
    assert army._census_due(clock[0] + 31)


def test_rule_defaults_take_the_first_offered_option_whose_classes_hold():
    book = playbook.load()
    assert book.rule('posture', ['gather', 'attack', 'defend'], {'base_threat': 'now', 'posture': 'gather'}) == 'defend'
    assert book.rule('posture', ['gather', 'attack', 'defend'], {'base_threat': 'now', 'posture': 'attack'}) == 'attack'
    assert book.rule('posture', ['gather', 'attack', 'defend'],
                     {'base_threat': 'now', 'posture': 'attack', 'raid': 'big'}) == 'defend'
    assert book.rule('posture', ['gather'], {'base_threat': 'now'}) == 'gather'
    assert book.rule('attack_size', book.options('attack_size'),
                     {'failed_attacks': 'none', 'game_time': 'late', 'enemy_defenses': 'strong'}) == 'larger'
    assert book.rule('intel', ['nothing', 'look_enemy_base', 'look_at_army'], {}) == 'look_at_army'


def test_fact_levels_pick_their_words_by_value():
    book = playbook.load()
    assert book.level('strategy', 'money', 5000)[0] == 'pile_up'
    assert book.level('strategy', 'money', 599)[0] == 'run_out'
    name, words = book.level('strategy', 'base_threat', 42.7)
    assert name == 'recent' and words.format(age=42) == 'enemy units were near our base 42 s ago'


def test_a_profile_without_an_option_the_player_acts_on_is_refused():
    data = copy.deepcopy(playbook.load().data)
    del data['questions']['posture']['options']['defend']
    with pytest.raises(ValueError, match='defend'):
        playbook.Playbook(data)
    data = copy.deepcopy(playbook.load().data)
    data['questions']['attack_size']['rules'].insert(0, {'choose': 'huge'})
    with pytest.raises(ValueError, match='huge'):
        playbook.Playbook(data)


def test_numbers_and_words_in_the_profile_change_the_play():
    data = copy.deepcopy(playbook.load().data)
    data['playbook']['units']['conscripts']['early'] = 1
    original_size = data['questions']['attack_size']['options']['large']['tanks']
    data['questions']['attack_size']['options']['large']['tanks'] = original_size - 1
    data['facts']['strategy']['economy']['no_refinery'] = 'no ore refinery at all'
    world, army, reflex, time = game(playbook.Playbook(data))
    world.set('credits', 1000)
    world.units_built.update(conscript=1, attack_dog=2)
    assert reflex.next_item('infantry', ('conscript',), time[0]) is None
    assert reflex.attack_size(time[0]) == original_size - 1
    assert reflex.strategy_facts(time[0])[0]['economy'] == 'no ore refinery at all'
    world, army, reflex, time = game()
    assert reflex.attack_size(time[0]) == original_size
    world.set('credits', 1000)
    world.units_built.update(conscript=1, attack_dog=2)
    assert reflex.next_item('infantry', ('conscript',), time[0]) == 'conscript'


def test_profile_controls_stored_attack_minimum_and_production_batch_size():
    book = playbook.load()
    world, army, reflex, time = game(book)
    army.intent.set('attack_size', 7, 'systemone')
    assert reflex.attack_size(time[0]) == 7
    book['attack']['early_min_above_small'] = 4
    assert reflex.attack_size(time[0]) == book.option('attack_size', 'large')['tanks']
    book['units']['queue_two']['count'] = 3
    world.set('credits', 4000)
    assert reflex.queue_count('rhino_tank') == 3
    world.set('credits', 1000)
    assert reflex.queue_count('rhino_tank') == 1


def test_profile_controls_army_census_refresh_and_freshness():
    book = playbook.load()
    world, army, reflex, time = game(book)
    book['army']['census'].update(every=8, fresh_within=12)
    army.census_tanks, army.census_at = 3, time[0]
    world.units_built['rhino_tank'] = 5
    time[0] += 8
    assert army._census_due(time[0]) and army.available() == 3
    time[0] += 5
    assert army.available() == 5
