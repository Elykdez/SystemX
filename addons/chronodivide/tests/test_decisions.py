"""Decision-check fixtures follow the selected profile and its JSON values."""

import copy
import json

from addons.chronodivide import playbook
from addons.chronodivide.tools.decisions import load_checks, scenario


def test_custom_profile_resolves_checks_relative_to_its_directory(tmp_path, monkeypatch):
    book, checks = load_checks()
    profile_dir = tmp_path / 'profile'
    profile_dir.mkdir()
    book.data['decision_checks'] = 'custom-checks.json'
    checks['defaults']['credits'] = 4321
    checks['defaults']['units']['rhino_tank'] = 11
    checks['radar_home'] = [920, 100]
    profile = profile_dir / 'custom.json'
    profile.write_text(json.dumps(book.data), encoding='utf-8')
    (profile_dir / 'custom-checks.json').write_text(json.dumps(checks), encoding='utf-8')
    (profile_dir / book.data['knowledge']).write_text(json.dumps(playbook.knowledge(book)), encoding='utf-8')
    monkeypatch.chdir(tmp_path)

    loaded_book, loaded_checks = load_checks(profile)
    world, reflex, army, now = scenario(loaded_checks, book=loaded_book)
    assert world.credits.value == 4321
    assert world.units_built['rhino_tank'] == 11
    assert army.ctx.radar_home == (920, 100)
    assert world.knowledge['faction'] == 'soviet'


def test_scenario_overrides_do_not_change_defaults_or_other_scenarios():
    book, checks = load_checks()
    original = copy.deepcopy(checks)
    world, _, _, _ = scenario(checks, book=book,
                              structures={'war_factory': None, 'ore_refinery': 1},
                              units={'rhino_tank': 7})
    assert 'war_factory' not in world.structures
    assert world.structures['ore_refinery'] == 1
    assert world.structures['construction_yard'] == 1
    assert world.units_built['rhino_tank'] == 7
    world.power.value['level'] = 'low'
    baseline, _, _, _ = scenario(checks, book=book)
    assert baseline.structures['war_factory'] == 1
    assert baseline.units_built['rhino_tank'] == 0
    assert checks == original


def test_soviet_economy_and_technology_fixtures_keep_implicit_units():
    book, checks = load_checks()
    cases = {case['name']: case for case in checks['cases']['production']}
    world, _, _, _ = scenario(checks, book=book, **cases['rich, battle lab']['scenario'])
    assert world.structures['battle_lab'] == world.structures['radar_tower'] == 1
    assert world.structures['ore_refinery'] + world.units_built['war_miner'] == 4
    world, _, _, _ = scenario(checks, book=book, **cases['dog scouts first']['scenario'])
    assert world.structures['ore_refinery'] + world.units_built['war_miner'] == 1
