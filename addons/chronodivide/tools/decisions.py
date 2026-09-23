"""System One decision check: game situations with known right answers.

    python -m addons.chronodivide.tools.decisions [--server URL] [--show] [--profile PATH]

Each scenario sets up the world (credits, base, army, threats, radar, failed attacks, what
the enemy base looked like) and builds the exact questions the player would send, through
the player's own code and the System One profile (Reflex.ask_strategy, Reflex.next_item). Needs
the System One server; prints per-question accuracy, how often the profile's rule default agrees,
and every miss. Run it after changing any wording in the profile.
The profile's decision_checks entry names the JSON fixture file, relative to that profile.

Results apply only to the model configuration and profile used by that run.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np

from systemone_client import SystemOneClient

from .. import playbook
from ..allied_tactics import choices as prism_choices, facts as prism_facts
from ..broker import Decision
from ..army import Army
from ..intel import Intel
from ..intent import StandingIntent
from ..layout import Layout
from ..perception.radar import Radar
from ..reflex import Reflex
from ..skills import Context
from ..world import World

class Capture:
    def __init__(self, clock):
        self.clock, self.submitted = clock, []

    def submit(self, decision):
        self.submitted.append(decision)

    def busy_keys(self):
        return set()


def load_checks(profile=playbook.PROFILE):
    """Resolve evaluation fixtures relative to the selected runtime profile."""
    book = playbook.load(profile)
    path = Path(profile).parent / book.data['decision_checks']
    checks = json.loads(path.read_text(encoding='utf-8'))
    if checks.get('version') != 1:
        raise ValueError('Unsupported decision-check configuration.')
    return book, checks


def scenario(checks, book=None, **overrides):
    """Build a world; optional census records measured strength, healthy vehicles and age."""
    settings = {**checks['defaults'], **overrides}
    now = [checks['clock']]
    world = World(knowledge=playbook.knowledge(book) if book else None, clock=lambda: now[0])
    world.started = now[0] - settings['minutes'] * 60
    world.set('credits', settings['credits'])
    world.set('power', dict(settings['power']))
    for section, target in (('structures', world.structures), ('units', world.units_built)):
        values = {**checks['defaults'][section], **overrides.get(section, {})}
        target.update({item: count for item, count in values.items() if count is not None})
    if settings['radar']:
        radar = dict(checks['radar'])
        for field in ('bounds', 'view', 'origin'):
            if radar[field] is not None:
                radar[field] = tuple(radar[field])
        if settings['enemy']:
            radar['enemy'] = checks['enemy_contacts']
        for field in ('own', 'enemy'):
            radar[field] = tuple((area, tuple(point)) for area, point in radar[field])
        world.set('radar', Radar(**radar))
    ctx = Context(Layout.for_canvas(*checks['canvas']), world)
    ctx.radar_home = tuple(checks['radar_home'])
    intent = StandingIntent(attack_size=settings['attack_size'])
    intel = Intel(lambda: now[0], book=book)
    army = Army(ctx, None, intent, clock=lambda: now[0], intel=intel, book=book)
    army.failures = settings['failures']
    army.scouts_ready, army.explored = settings['scouts'], settings['explored']
    census = settings.get('census')
    if census is not None:
        army.census_tanks = census['tanks']
        army.census_healthy_vehicles = census['healthy_vehicles']
        army.census_at = now[0] - census['age']
    if settings['enemy']:
        intel.update(world.radar.value, ctx.radar_home, now[0])
    for kind, age in (('enemy_base', settings['looked_base']), ('army', settings['looked_army'])):
        if age is not None:
            intel.record_look(kind, tuple(checks['look']['point']), dict(checks['look']['summary']), now[0] - age)
    if settings['strength'] is not None:
        intel.strength = settings['strength']
    if settings['scout'] is not None:
        intel.send_scout(tuple(checks['scout_target']), now[0] - settings['scout'])
    if settings['threat'] is not None:
        army.threat_at = now[0] - settings['threat']
        army.seen = (tuple(checks['threat']['point']), checks['threat']['pixels'])
    if settings['campaign']:
        campaign = checks['campaign']
        army.campaign.active, army.campaign.started = True, now[0] - settings['campaign']
        army.campaign.waves = campaign['waves']
        army.campaign.last_wave, army.campaign.kind = now[0] - campaign['last_wave_age'], campaign['kind']
        intent.posture = campaign['posture']
    reflex = Reflex(ctx, None, Capture(lambda: now[0]), intent, clock=lambda: now[0], army=army, book=book)
    reflex.last_strategy = -float('inf')
    return world, reflex, army, now


def ask(client, state, questions):
    started = time.perf_counter()
    result = client.decide(state, questions)
    return result['answers'], (time.perf_counter() - started) * 1000


class Tally:
    def __init__(self):
        self.score, self.misses, self.latencies = {}, [], []

    def add(self, name, qid, choice, expected, probabilities=None):
        right = self.score.setdefault(qid, [0, 0])
        right[1] += 1
        right[0] += choice in expected
        if choice not in expected:
            self.misses.append((name, qid, choice, sorted(expected), probabilities))

    def check(self, client, name, decision, expected, show):
        """Send one decision to System One; score its answer and the profile's rule default."""
        answers, ms = ask(client, decision.state, decision.questions)
        self.latencies.append(ms)
        for qid, answer in answers.items():
            if qid not in expected:
                continue
            self.add(name, qid, answer['choice'], expected[qid], answer.get('probabilities'))
            self.add(name, f'{qid} (rule)', decision.default[qid]['choice'], expected[qid])
        if show:
            print(json.dumps(dict(scenario=name, state=decision.state, answers=answers), indent=1))


def asked(reflex, qid):
    return next((d for d in reflex.broker.submitted if qid in d.questions), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    parser.add_argument('--profile', default=str(playbook.PROFILE), help='System One profile to check.')
    parser.add_argument('--show', action='store_true', help='Print every state and answer.')
    args = parser.parse_args()
    book, checks = load_checks(args.profile)
    tally = Tally()
    with SystemOneClient(args.server, timeout=10) as client:
        client.wait_ready(timeout=180)
        for case in checks['cases']['strategy']:
            name, kwargs = case['name'], case['scenario']
            world, reflex, army, now = scenario(checks, book=book, **kwargs)
            if kwargs.get('threat') is not None:
                world.note('attack', near=list(checks['threat']['point']), pixels=checks['threat']['pixels'])
            reflex.ask_strategy(now[0])
            for qid, expected in case['expected'].items():
                if qid == 'posture' and 'focus' in book.questions:
                    qid = 'focus'
                    expected = ['build' if option == 'gather' else option for option in expected]
                decision = asked(reflex, qid)
                if decision is not None:
                    tally.check(client, name, decision, {qid: expected}, args.show)
                else:
                    tally.add(name, f'{qid} (rule only)', getattr(reflex.intent, qid), expected)
        for case in checks['cases']['attack_size']:
            name, kwargs, expected = case['name'], case['scenario'], case['expected']
            world, reflex, army, now = scenario(checks, book=book, **kwargs)
            reflex.ask_strategy(now[0])
            decision = asked(reflex, 'attack_size')
            if decision is None:
                size = reflex.attack_size(now[0])
                choice = next((option for option in book.options('attack_size')
                               if book.option('attack_size', option)['tanks'] == size), None)
                tally.add(name, 'attack_size (rule only)', choice, expected)
                continue
            tally.check(client, name, decision, {'attack_size': expected}, args.show)
        for case in checks['cases']['intel']:
            name, kwargs, expected = case['name'], case['scenario'], case['expected']
            world, reflex, army, now = scenario(checks, book=book, **kwargs)
            reflex.ask_strategy(now[0])
            decision = asked(reflex, 'intel')
            if decision is None or len(decision.questions['intel']['criteria']) < 2:
                tally.add(name, 'intel (not asked)', None, expected)
                continue
            tally.check(client, name, decision, {'intel': expected}, args.show)
        for case in checks['cases']['production']:
            name, kwargs, expected = case['name'], case['scenario'], case['expected']
            lane, available = case['lane'], case['available']
            world, reflex, army, now = scenario(checks, book=book, **kwargs)
            default = reflex.next_item(lane, available, now[0])
            if default is not None or not reflex.broker.submitted:     # one sensible option (or none): no question
                tally.add(name, 'production (rule only)', default, expected)
                continue
            tally.check(client, name, reflex.broker.submitted[-1], {'production': expected}, args.show)
        for case in checks['cases'].get('resources', ()):
            name, expected = case['name'], case['expected']
            world, reflex, army, now = scenario(checks, book=book, **case['scenario'])
            evidence, state = case['miner_evidence'], army.miners
            state.grouped = evidence['saved']
            state.home_ore_chips = evidence['home_chips']
            state.home_ore_seen = state.home_ore_chips >= book['resources']['home_min_chips']
            state.home_ore_at = now[0] - evidence['home_age']
            state.home_ore_target = tuple(evidence['home_target']) if evidence.get('home_target') else None
            if evidence.get('income_idle_for') is not None:
                state.funding_at = now[0] - evidence['income_idle_for']
                state.income_at = now[0]
            state.home_ore_thin_since = (None if evidence['thin_for'] is None else
                                         now[0] - evidence['thin_for'])
            if evidence['gem_age'] is not None:
                state.field, state.field_chips = tuple(checks['radar_home']), 15
                state.field_at = now[0] - evidence['gem_age']
            if world.radar.value is not None:
                width, height = checks['canvas']
                world.set('radar', replace(world.radar.value,
                          explored=np.full((height, width), evidence['explored'], dtype=bool)))
            reflex.ask_strategy(now[0])
            decision = asked(reflex, 'resources')
            if decision is None:
                tally.add(name, 'resources (rule only)', reflex.intent.resources, expected)
            else:
                tally.check(client, name, decision, {'resources': expected}, args.show)
        for case in checks['cases'].get('placement', ()):
            world, reflex, army, now = scenario(checks, book=book, **case['scenario'])
            reflex.ctx.build_off_ally = case['build_off_ally']
            reflex.intent.focus = case.get('focus', 'build')
            radar = world.radar.value
            if radar is not None:
                world.set('radar', replace(radar, allies=((20, (940, 92)),),
                          enemy=((4, (941, 92)),) if case.get('enemy_at_ally') else radar.enemy))
            reflex.placement_target(case['item'], 0, now[0])
            decision = asked(reflex, 'placement')
            if decision is None:
                tally.add(case['name'], 'placement (rule only)', reflex.placement_choice, case['expected'])
            else:
                tally.check(client, case['name'], decision, {'placement': case['expected']}, args.show)
        for case in checks['cases'].get('prism_battle', ()):
            classes = prism_facts(case['view'], case['clear'])
            options = prism_choices(case['view'], case['clear'])
            choice = book.rule('prism_battle', options, classes)
            if len(options) == 1:
                tally.add(case['name'], 'prism_battle (rule only)', choice, case['expected'])
                continue
            state = {key: book.say('prism_battle', key, value) for key, value in classes.items()}
            decision = Decision('tactical:prism_battle', 'tactical', state, book.question('prism_battle', options),
                                0, None, {'prism_battle': {'choice': choice}})
            tally.check(client, case['name'], decision, {'prism_battle': case['expected']}, args.show)
    print('accuracy:', {k: f'{a}/{n}' for k, (a, n) in sorted(tally.score.items())})
    latencies = sorted(tally.latencies)
    print(f'latency ms (client, per request): median {latencies[len(latencies) // 2]:.0f}, max {latencies[-1]:.0f}')
    for name, qid, choice, expected, probabilities in tally.misses:
        top = sorted((probabilities or {}).items(), key=lambda kv: -kv[1])[:3]
        print(f'  miss: {name} [{qid}] chose {choice}, expected {expected}; top {[(k, round(v, 2)) for k, v in top]}')


if __name__ == '__main__':
    main()
