"""Fast deterministic policy that executes standing intent; System One fills in bounded choices.

Each tick the reflex looks at the world and, if the sidebar is free, starts at most
one sidebar task: place a ready building, queue the next item in an idle lane, or
refresh a lane it has not seen recently. Choices with several sensible options
are asked of System One through the broker with a deadline; a rule default is always
ready, so the player never waits for the model.

Strategy (every 15 s, and at once when enemies show up near the base) asks System One up to
four short questions, as separate requests that run in parallel on the server's workers:
what the money is spent on (objective), what the army does (posture), how many tanks the
army needs at home before it attacks (attack size), and what to find out about the
battlefield next (intel: send a scout, look at the enemy base, look at our fight). Only
feasible options are offered, and each question's state holds just the facts its criteria
refer to, in the same words; any other fact measurably confused the model.

How to play comes from the System One profile (playbook.py): the questions and their words, the
words of the facts, the rule defaults, the build order and every number of the production
rules. This module measures the facts, works out what is feasible now, and queues and
places what was decided.
"""

import time

from . import miners, playbook, skills
from .army import combat_units
from .broker import Decision
from .perception.radar import distance

LANE_UNLOCK = {'buildings': 'construction_yard', 'defenses': 'barracks', 'infantry': 'barracks',
               'vehicles': 'war_factory'}
UNIT_LANES = ('infantry', 'vehicles')
PLACE_ATTEMPTS = 6              # failed placements before a finished building is cancelled (refund)
SKIP_AFTER_CANCEL = 300.0       # seconds a cancelled building stays out of the build order
POWER_GRACE = 10.0              # seconds the power bar takes to show a newly placed reactor


class Reflex:
    def __init__(self, ctx, runner=None, broker=None, intent=None, log=None, clock=time.monotonic,
                 lane_refresh=2.5, generation=lambda: 0, army=None, book=None):
        self.ctx, self.runner, self.broker, self.intent = ctx, runner, broker, intent
        self.log, self.clock = log, clock
        self.book = book or playbook.default()
        self.lane_refresh = lane_refresh
        self.generation = generation
        self.army = army
        self.choices = {}              # lane -> item chosen (by System One or default), consumed when queued
        self.last_strategy = -1e9
        self.last_focus_check = -1e9
        self.cooldown = {}             # task name -> time before which it is not retried
        self.failures = {}             # task name -> failure count
        self.hold = {}                 # lane -> time before which nothing new is queued there
        self.recommendation = None
        self.awaited = None            # (item, since): next build-order item not yet available
        self.skip = {}                 # item -> time until which the build order leaves it out
        self.power_grace = -1e9        # the power bar fills slowly after a reactor is placed
        self.placement_event = None
        self.placement_choice = None
        if army is not None:
            army.required_size = self.attack_size
            army.broker = broker

    # -- facts ---------------------------------------------------------------
    def economy(self):
        world, plan = self.ctx.world, self.book['units']['miners']
        refineries = world.structures.get('ore_refinery', 0)
        miners = refineries + world.units_built.get(plan['item'], 0)       # each refinery comes with one
        expanding = self.intent is not None and self.intent.objective == 'expand_economy'
        extra = plan['extra_when_expanding'] if expanding else 0
        wanted = min(plan['max'], plan['per_refinery'] * refineries + extra) if refineries else 0
        return refineries, miners, wanted

    def tank_count(self):
        built = self.ctx.world.units_built
        return sum(built.get(item, 0) for item in self.book['units']['tanks'])

    def tanks(self, count, until, now):
        """At least `count` tanks were built (or the game is older than `until` seconds)."""
        return self.tank_count() >= count or (until is not None and now - self.ctx.world.started >= until)

    def threatened(self, within, now):
        """Enemies were near the base in the last `within` seconds."""
        return self.army is not None and self.army.threat_age(now) <= within

    def attack_size(self, now):
        """Tanks needed at home before an attack: System One's answer, or the rule default meanwhile."""
        options = self.feasible_attack_sizes(now)
        minimum = min(self.book.option('attack_size', option)['tanks'] for option in options)
        facts = self.attack_size_facts(now)[1]
        if (facts['game_time'] == 'early' and facts['failed_attacks'] == 'none'
                and facts['enemy_defenses'] != 'strong'):
            minimum = self.book.option('attack_size', 'small')['tanks'] + self.book['attack']['early_min_above_small']
        if self.intent is not None and self.intent.attack_size is not None and self.intent.attack_size >= minimum:
            return self.intent.attack_size
        choice = self.book.rule('attack_size', options, facts)
        return self.book.option('attack_size', choice)['tanks']

    def feasible_attack_sizes(self, now):
        """A small cleanup force needs observed weak defenses; elapsed time alone proves nothing."""
        facts = self.attack_size_facts(now)[1]
        options = self.book.options('attack_size')
        if (facts['failed_attacks'] != 'none' or facts['game_time'] == 'late'
                or facts['enemy_defenses'] == 'strong'):
            choice = self.book.rule('attack_size', options, facts)
            minimum = self.book.option('attack_size', choice)['tanks']
            return [option for option in options if self.book.option('attack_size', option)['tanks'] >= minimum]
        return [option for option in options if option != 'small']

    def strategy_facts(self, now):
        """(words, classes): the strategy state in the words the criteria use, and each fact's
        class for the rule defaults (with the current posture and the raid size)."""
        world, army, book = self.ctx.world, self.army, self.book

        def say(fact, key, **values):
            return book.say('strategy', fact, key, **values)
        refineries, miners, wanted = self.economy()
        if refineries < 2:
            economy = 'one_refinery' if refineries == 1 else 'no_refinery'
        else:
            economy = 'few_miners' if miners < wanted else 'enough'
        credits = world.spendable()
        if credits is None:
            money, money_words = 'unknown', say('money', 'unknown')
        else:
            money, level = book.level('strategy', 'money', credits)
            money_words = say('money', 'text', credits=credits, level=level)
        size = self.attack_size(now)
        available = army.available() if army is not None else combat_units(world, book)
        reached = 'reached' if available >= size else 'missing'
        campaign = army.campaign if army is not None else None
        if campaign is None or not campaign.active:
            attack, attack_words = 'not_started', say('attack', 'not_started')
        else:
            targets = book.words['strategy']['attack']['targets']
            attack = 'under_way'
            attack_words = say('attack', 'under_way', seconds=int(now - campaign.started), waves=campaign.waves,
                               last=int(now - campaign.last_wave), target=targets.get(campaign.kind, targets['none']))
        if army is None:
            threat, threat_words = 'unknown', say('base_threat', 'unknown')
        else:
            age = army.threat_age(now)
            threat, template = book.level('strategy', 'base_threat', age)
            threat_words = template.format(age=int(age))
        words = {'economy': say('economy', economy), 'money': money_words,
                 'army': say('army', 'text', available=available, size=size,
                             status=say('army', reached, missing=size - available)),
                 'attack': attack_words, 'base_threat': threat_words}
        classes = {'economy': economy, 'money': money, 'army': reached, 'attack': attack, 'base_threat': threat,
                   'posture': self.intent.posture if self.intent else 'gather',
                   'raid': army.raid() if army is not None else None}
        return words, classes

    def attack_size_facts(self, now):
        """(words, classes) for the attack size question: game time, failed attacks, and how
        strong the enemy base looked, and whether reinforcements have stalled."""
        world, army = self.ctx.world, self.army
        played = now - world.started
        phase = 'late' if played >= self.book['attack']['late_game'] else 'early'
        failures = army.failures if army is not None else 0
        failed = 'none' if failures == 0 else 'one' if failures == 1 else 'more'
        intel = army.intel if army is not None else None
        defenses = (intel.strength if intel is not None else None) or 'unknown'
        plan = self.book['resources']
        credits = world.spendable()
        stalled = (army is not None and credits is not None and credits < plan['low_credits_below']
                   and miners.income_stalled(army.miners, plan, now)
                   and now - army.census_at <= self.book['army']['census']['fresh_within']
                   and (army.census_tanks or 0) >= self.book.option('attack_size', 'large')['tanks'])
        reinforcements = 'stalled' if stalled else 'not_limited'
        say = self.book.say
        words = {'game_time': say('attack_size', 'game_time', phase, minutes=int(played // 60)),
                 'failed_attacks': say('attack_size', 'failed_attacks', failed, count=failures),
                 'enemy_defenses': say('attack_size', 'enemy_defenses', defenses),
                 'reinforcements': say('attack_size', 'reinforcements', reinforcements)}
        return words, {'game_time': phase, 'failed_attacks': failed, 'enemy_defenses': defenses,
                       'reinforcements': reinforcements}

    def intel_facts(self, now):
        """The battlefield picture in words (the words the intel question's criteria use)."""
        intel, active = self.army.intel, self.army.campaign.active

        def say(fact, key, **values):
            return self.book.say('intel', fact, key, **values)
        if intel.enemy_base is None:
            base = say('enemy_base', 'not_found')
        else:
            age = intel.looked('enemy_base', now)
            defenses = say('enemy_base', 'defenses', strength=intel.strength) if intel.strength else ''
            if age is None:
                base = say('enemy_base', 'stale')
            else:
                recent = 'recent' if not intel.stale('enemy_base', now) else 'stale_seen'
                base = say('enemy_base', recent, age=int(age), defenses=defenses)
        scout = (say('scout', 'out', age=int(now - intel.scout['at'])) if intel.scout_out(now)
                 else say('scout', 'none'))
        if not active:
            attack = say('attack', 'not_started')
        else:
            look = intel.looks.get('army')
            detail = (say('attack', 'detail', age=int(now - look['at']), own=look['own_objects'],
                          enemy=look['enemy_objects']) if look else '')
            attack = say('attack', 'recent' if not intel.stale('army', now) else 'stale') + detail
        surroundings = say('surroundings', 'explored' if self.army.explored else
                           'unexplored' if self.army.scouts_ready else 'no_scouts')
        return {'enemy_base': base, 'scout': scout, 'attack': attack, 'surroundings': surroundings}

    # -- strategy --------------------------------------------------------------
    def high_tech_ready(self, now, funded=True):
        """Surplus economy plus fresh evidence of an advantage, never cumulative tanks alone."""
        plan, army = self.book['high_tech'], self.army
        refineries, miners, _ = self.economy()
        if ((funded and (self.ctx.world.spendable() or 0) < plan['min_credits']) or
                refineries < plan['min_refineries'] or miners < plan['min_miners'] or
                army is None or (plan.get('require_enemy_advantage', True) and army.intel is None) or
                self.threatened(plan['quiet_seconds'], now)):
            return False
        if (army.census_tanks is None or now - army.census_at > plan['fresh_within'] or
                army.census_tanks < plan['min_army'] or
                army.census_healthy_vehicles < plan['min_healthy_vehicles']):
            return False
        if not plan.get('require_enemy_advantage', True):
            return True
        look = army.intel.looks.get('enemy_base')
        return bool(look and army.intel.enemy_base is not None and
                    now - look['at'] <= plan['fresh_within'] and
                    look['shroud'] <= plan['max_shroud'] and look['enemy_objects'] > 0 and
                    army.intel.strength in plan['enemy_strength'])

    def feasible_objectives(self):
        """Only objectives that change something now are offered to System One."""
        world = self.ctx.world
        refineries, miners, wanted = self.economy()
        options = ['build_army']
        if refineries < 2 or miners < wanted:
            options.append('expand_economy')
        if (world.structures.get('war_factory') and not world.structures.get('battle_lab') and
                self.high_tech_ready(self.clock())):
            options.append('tech_up')
        return options

    def feasible_postures(self, now):
        options = ['gather']
        army = self.army
        if army is None:
            return options
        threat = self.book.level('strategy', 'base_threat', army.threat_age(now))[0]
        if threat == 'now' and (not army.campaign.active or army.raid() == 'big'):
            return ['defend']
        radar = self.ctx.world.radar.value
        ranged_ready = not self.book['battle'].get('prism') or army.prism_plan() is not None
        if ranged_ready and (army.campaign.active or
                             (radar is not None and radar.online and army.available() >= self.attack_size(now))):
            options.append('attack')
        if self.threatened(self.book['defense']['offer_within'], now):
            options.append('defend')
        return options

    def feasible_intel(self, now):
        """Exploring needs verified dog scouts; scouting also needs radar. Looks need a target."""
        options = ['nothing']
        army = self.army
        intel = army.intel if army is not None else None
        if intel is None:
            return options
        if (army.scouts_ready and not army.explored and not army.exploring and not army.explore_failed
                and not army.campaign.active):
            options.append('explore')
        radar = self.ctx.world.radar.value
        if radar is None or not radar.online:
            return options
        if intel.enemy_base is None and not intel.scout_out(now) and not army.campaign.active and \
                army.scouts_ready:
            options.append('scout')
        if intel.enemy_base is not None and intel.stale('enemy_base', now):
            options.append('look_enemy_base')
        if army.campaign.active and intel.stale('army', now):
            options.append('look_at_army')
        return options

    def resource_facts(self, now):
        state, plan = self.army.miners, self.book['resources']
        say = self.book.say
        if now - state.home_ore_at > plan['home_fresh_within']:
            ore = 'unknown'
        elif miners.home_thin(state, plan, now):
            ore = 'thin'
        elif state.home_ore_seen:
            ore = 'present'
        else:
            ore = 'unknown'
        credits = self.ctx.world.spendable()
        money = ('unknown' if credits is None else
                 'low' if credits < plan['low_credits_below'] else 'enough')
        field = ('unknown' if state.field is None else
                 'recent' if miners.field_fresh(state, plan, now) else 'stale')
        team = 'saved' if state.grouped else 'missing'
        income = ('unknown' if state.funding_at is None or
                  now - state.income_at > plan['home_fresh_within'] else
                  'stalled' if miners.income_stalled(state, plan, now) else 'progressing')
        threat = 'recent' if self.threatened(plan['quiet_seconds'], now) else 'quiet'
        words = {
            'home_ore': say('resources', 'home_ore', ore, chips=state.home_ore_chips),
            'money': say('resources', 'money', money, credits=credits),
            'gem_field': say('resources', 'gem_field', field,
                             age=int(now - state.field_at), chips=state.field_chips),
            'miner_team': say('resources', 'miner_team', team),
            'income': say('resources', 'income', income, seconds=plan['income_stall_seconds']),
            'base_threat': say('resources', 'base_threat', threat),
        }
        return words, dict(home_ore=ore, money=money, gem_field=field,
                           miner_team=team, income=income, base_threat=threat)

    def feasible_resources(self, now):
        options = ['keep_home']
        army = self.army
        if army is None:
            return options
        state, plan = army.miners, self.book['resources']
        facts = self.resource_facts(now)[1]
        if (plan['enabled'] and state.grouped and not state.engaged and
                state.resource_phase == 'idle' and not army.campaign.active and
                now - state.resource_last >= plan['retry_after'] and
                now - state.home_ore_at <= plan['home_fresh_within'] and
                facts['home_ore'] in ('present', 'thin') and
                facts['money'] == 'low' and facts['base_threat'] == 'quiet'):
            if miners.income_stalled(state, plan, now) and miners.home_ore_ready(state, plan, now):
                options.append('harvest_home_ore')
            if facts['home_ore'] == 'thin' and miners.field_fresh(state, plan, now):
                options.append('harvest_gems')
            elif (facts['home_ore'] == 'thin' and now - state.survey_at >= plan['survey_every'] and
                  miners.survey_point(army) is not None):
                options.append('survey_gems')
        return options

    def _strategy_result(self, decision, answers, provenance, meta):
        if not answers or not self.intent:
            return
        questions = self.book.questions
        focus = answers.get('focus', {}).get('choice')
        if focus in ('attack', 'defend', 'build'):
            # A late answer must not overwrite a newer incident or command an infeasible posture.
            _, event = self.focus_facts(self.clock())
            if event == self.intent.focus_event:
                posture = 'gather' if focus == 'build' else focus
                if posture in self.feasible_postures(self.clock()):
                    self.intent.set('focus', focus, provenance)
                    self.intent.set('posture', posture, provenance)
                    self.ctx.world.note('ai_focus', focus=focus, incident=list(event), source=provenance)
        for qid in ('objective', 'posture'):
            choice = answers.get(qid, {}).get('choice')
            feasible = self.feasible_postures(self.clock()) if qid == 'posture' else self.feasible_objectives()
            if choice in feasible:
                self.intent.set(qid, choice, provenance)
        size = answers.get('attack_size', {}).get('choice')
        if size in self.feasible_attack_sizes(self.clock()):
            self.intent.set('attack_size', self.book.option('attack_size', size)['tanks'], provenance)
        intel = answers.get('intel', {}).get('choice')
        if intel in questions['intel']['options']:
            self.intent.set('intel', None if intel == 'nothing' else intel, provenance)
        resources = answers.get('resources', {}).get('choice')
        if resources in self.feasible_resources(self.clock()):
            self.intent.set('resources', resources, provenance)

    def focus_facts(self, now):
        postures = self.feasible_postures(now)
        phase = 'defend' if postures == ['defend'] else 'attack' if 'attack' in postures else 'build'
        campaign = self.army.campaign if self.army else None
        event = (phase, campaign.started if campaign and campaign.active else None,
                 self.army.failures if self.army else 0, 'defend' in postures)
        return self.strategy_facts(now)[0], event

    def ask_focus(self, now):
        if 'focus' not in self.book.questions:
            return
        if now - self.last_focus_check < self.book['focus']['check_every']:
            return
        self.last_focus_check = now
        facts, event = self.focus_facts(now)
        if self.intent.focus_event == event:
            return
        self.intent.set('focus_event', event, 'incident')
        options = ['build' if option == 'gather' else option for option in self.feasible_postures(now)]
        default = self.book.rule('focus', options, {'situation': event[0]})
        self.intent.set('focus', default, 'rule')
        self.intent.set('posture', 'gather' if default == 'build' else default, 'rule')
        self.ctx.world.note('ai_focus', focus=default, incident=list(event), source='rule')
        if self.broker is not None and len(options) > 1:
            self.broker.submit(Decision('strategy:focus', 'urgent', self.book.state('focus', facts),
                              self.book.question('focus', options),
                              self.broker.clock() + self.book.systemone['deadlines'].get(
                                  'focus', self.book.systemone['deadlines']['strategy']),
                              lambda decision, answers, provenance, meta:
                              self._strategy_result(decision, answers, provenance, meta)
                              if self.intent.focus_event == event else None,
                              {'focus': {'type': 'choice', 'choice': default}}, generation=self.generation()))

    def ask_strategy(self, now):
        if self.intent is None:
            return
        book, systemone = self.book, self.book.systemone
        self.ask_focus(now)
        objectives = self.feasible_objectives()
        postures = self.feasible_postures(now)
        # Standing intent never stays on an option that is no longer feasible.
        if self.intent.objective not in objectives or self.intent.posture not in postures:
            classes = self.strategy_facts(now)[1]
            if self.intent.objective not in objectives:
                self.intent.set('objective', book.rule('objective', objectives, classes), 'rule')
            if self.intent.posture not in postures:
                self.intent.set('posture', book.rule('posture', postures, classes), 'rule')
        resources = self.feasible_resources(now)
        if self.intent.resources not in resources:
            self.intent.set('resources', 'keep_home', 'rule')
        urgent = (self.army is not None and self.army.threat_age(now) < systemone['urgent']['threat_within']
                  and now - self.last_strategy > systemone['urgent']['gap'])
        if self.broker is None or (now - self.last_strategy < systemone['strategy_every'] and not urgent):
            return
        self.last_strategy = now
        facts, classes = self.strategy_facts(now)
        # One focused request per question: (options, the facts its criteria are worded against, classes).
        asks = {'objective': (objectives, facts, classes)}
        if 'focus' not in book.questions:
            asks['posture'] = (postures, facts, classes)
        if self.army is not None and not self.army.campaign.active and not urgent:
            asks['attack_size'] = (self.feasible_attack_sizes(now), *self.attack_size_facts(now))
        if self.army is not None and self.army.intel is not None:
            asks['intel'] = (self.feasible_intel(now), self.intel_facts(now), {})
        if self.army is not None:
            asks['resources'] = (resources, *self.resource_facts(now))
        kind = 'urgent' if urgent else 'strategy'
        for qid, (options, state, rule_classes) in asks.items():
            if len(options) < 2:
                continue                      # a single option needs no question
            default = {qid: {'type': 'choice', 'choice': book.rule(qid, options, rule_classes)}}
            self.broker.submit(Decision(f'strategy:{qid}', kind, book.state(qid, state), book.question(qid, options),
                                        self.broker.clock() + systemone['deadlines'][kind], self._strategy_result,
                                        default, generation=self.generation()))

    # -- production ----------------------------------------------------------------
    def _decided(self, lane):
        def callback(decision, answers, provenance, meta):
            choice = (answers or {}).get('production', {}).get('choice')
            if choice:
                self.choices[lane] = (choice, provenance)
                if self.intent:
                    self.intent.set('production', choice, provenance, key=lane)
        return callback

    def unit_options(self, lane, available, spendable, now):
        """(options in rule-priority order, needs in words) for a unit or defense lane."""
        world, units = self.ctx.world, self.book['units']
        built = world.units_built
        options, needs = [], []

        def need(key, **values):
            needs.append(self.book.say('production', 'needs', key, **values))

        def can(item):
            return item in available and world.item(item)['cost'] <= spendable
        if lane == 'vehicles':
            radar = world.knowledge.get('radar_loss')
            if (radar and world.has('construction_yard') and world.has('battle_lab') and not world.has(radar['building']) and
                    world.radar.value is not None and not world.radar.value.online):
                tank = units['tank']
                # A fresh raid still needs a defender while radar funding recovers.
                return ([tank] if self.threatened(self.book['defense']['offer_within'], now) and can(tank)
                        else []), needs
            anti_air = units['anti_air']
            if (self.tank_count() >= anti_air['after_tanks'] and not built.get(anti_air['item'], 0)
                    and anti_air['item'] in available):
                # Rhinos cannot answer aircraft. Cover the first tanks before saving for expansion.
                need('anti_air', flak=0, tanks=self.tank_count())
                return ([anti_air['item']] if can(anti_air['item']) else []), needs
            recon = units.get('reconnaissance')
            if (recon and self.reconnaissance_due(world.lanes['defenses'].available) and
                    not self.threatened(recon['save_unless_threat_within'], now)):
                need('reconnaissance')
                return [], needs              # reveal the approach before funding more siege units
            refineries, miners, wanted = self.economy()
            plan = units['miners']
            miner_due = (miners < wanted and self.tanks(plan['after_tanks'], plan['tanks_first_until'], now) and
                         not self.threatened(plan['save_unless_threat_within'], now))
            if (self.saving_for_refinery(spendable, now) and
                    not (plan.get('before_second_refinery') and miner_due)):
                need('saving_refinery')
                return [], needs
            if miner_due:
                need('miners', miners=miners, wanted=wanted)
                if can(plan['item']):
                    options.append(plan['item'])
                elif plan['item'] in available:
                    return [], needs          # save up for the miner (see the profile)
            tech = self.book['high_tech']
            if (tech.get('save_after_tanks') is not None and not world.has('battle_lab') and
                    world.has('airforce_command') and self.tank_count() >= tech['save_after_tanks'] and
                    self.high_tech_ready(now, funded=False)):
                return [], needs              # fund Allied siege technology before more basic tanks
            siege = units['siege']
            siege_limit = self._siege_limit()
            siege_due = (self.army is not None and self.army.failures >= siege.get('min_failures', 1) and
                         self.tank_count() >= siege['after_tanks'] and
                         built.get(units['heavy_tank']['item'], 0) >= siege.get('min_heavy', 0) and
                         built.get(units['anti_air']['item'], 0) >= siege['min_flak'] and
                         built.get(siege['item'], 0) < siege_limit and
                         not self.threatened(siege['quiet_seconds'], now) and
                         (can(siege['item']) or (siege.get('before_heavy') and siege['item'] in available)))
            heavy = units['heavy_tank']
            advanced = (world.has('battle_lab') if heavy.get('after_lab') else self.high_tech_ready(now))
            if (spendable >= heavy['credits'] and advanced and
                    not (siege_due and siege.get('before_heavy'))):
                need('spare_credits', credits=spendable)
                if can(heavy['item']):
                    options.append(heavy['item'])
            plan = units['anti_air']
            tanks, flak = self.tank_count(), built.get(plan['item'], 0)
            if tanks >= plan['after_tanks'] and flak * plan['one_per_tanks'] < tanks:
                need('anti_air', flak=flak, tanks=tanks)
                if can(plan['item']):
                    options.append(plan['item'])
            plan = units['siege']
            if siege_due:
                need('siege')
                if can(plan['item']):
                    options.append(plan['item'])
                else:
                    return options, needs      # save for first-wave siege instead of another basic tank
            if (heavy.get('minimum', 0) > built.get(heavy['item'], 0) and world.has('battle_lab') and
                    heavy['item'] in available and not siege_due and
                    not self.threatened(siege['quiet_seconds'], now)):
                return options, needs          # finish the Allied mixed force before basic replacements
            if can(units['tank']):
                options.append(units['tank'])
        elif lane == 'infantry':
            plan = units['scouts']
            dogs = built.get(plan['item'], 0)
            if dogs < plan['count']:
                need('scouts', dogs=dogs, count=plan['count'])
                if can(plan['item']):
                    options.append(plan['item'])
                return options, needs         # a briefly unreadable dog cameo must not admit other infantry
            elif self.army is not None and self.army.holding_for_scouts(now):
                return options, needs         # other infantry waits: P must select the dogs alone
            guards = units.get('base_dogs')
            if guards and world.has('war_factory') and dogs < units['scouts']['count'] + guards['count']:
                reserve = guards['reserve_tanks'] * world.item(units['tank'])['cost']
                if spendable >= world.item(plan['item'])['cost'] + reserve and can(plan['item']):
                    options.append(plan['item'])
                    return options, needs
            plan = units['anti_air_infantry']
            tanks, flak = self.tank_count(), built.get(plan['item'], 0)
            if (tanks >= plan['after_tanks'] and flak < plan['max'] and
                    flak * plan['one_per_tanks'] < tanks and spendable >= plan['credits'] and can(plan['item'])):
                need('anti_air_infantry', flak=flak, tanks=tanks)
                options.append(plan['item'])
            plan = units['conscripts']
            conscripts = built.get(plan['item'], 0)
            if conscripts < plan['early']:
                need('conscripts', conscripts=conscripts)
                if can(plan['item']):
                    options.append(plan['item'])
            plan = units['troopers']
            if spendable >= plan['credits'] and self.high_tech_ready(now) and world.structures.get('war_factory') and \
                    self.tanks(plan['after_tanks'], None, now):
                need('spare_credits', credits=spendable)
                for item in plan['items']:
                    if can(item) and item not in options:
                        options.append(item)
        elif lane == 'defenses':
            recon = units.get('reconnaissance')
            if (self.reconnaissance_due(available) and
                    spendable >= world.item(recon['item'])['cost'] +
                    recon['reserve_tanks'] * world.item(units['tank'])['cost'] and can(recon['item'])):
                need('reconnaissance')
                return [recon['item']], needs
            plan = units['defenses']
            if self.basic_defense_due(now, spendable):
                need('defenses')
                if can(plan['item']):
                    options.append(plan['item'])
            fort = units.get('fortification')
            if fort and self.fortification_due(now, spendable) and can(fort['item']):
                need('fortification')
                options.append(fort['item'])
        return options, needs

    def reconnaissance_due(self, available=None):
        world, plan = self.ctx.world, self.book['units'].get('reconnaissance')
        return bool(plan and self.army is not None and self.army.intel is not None and
                    self.army.intel.enemy_base is None and not world.has(plan['item']) and
                    all(world.has(item) for item in plan['requires']) and
                    world.units_built.get(self.book['units']['heavy_tank']['item'], 0) >= plan.get('after_heavy', 0) and
                    (world.power.value or {}).get('level') == 'ok' and
                    (available is None or plan['item'] in available))

    def fortification_due(self, now, spendable=None):
        world, units = self.ctx.world, self.book['units']
        plan = units.get('fortification')
        spendable = world.spendable() if spendable is None else spendable
        return bool(plan and spendable is not None and all(world.has(i) for i in plan['requires']) and
                    self.tank_count() >= plan['after_tanks'] and
                    world.units_built.get(units['heavy_tank']['item'], 0) >= plan.get('after_heavy', 0) and
                    world.structures.get(plan['item'], 0) < plan['max'] and
                    (world.power.value or {}).get('level') == 'ok' and
                    not self.threatened(plan['quiet_seconds'], now) and
                    spendable >= world.item(plan['item'])['cost'] +
                    plan['reserve_tanks'] * world.item(units['tank'])['cost'])

    def basic_defense_due(self, now, spendable=None):
        """A little static cover after the mobile force stands, without spending its next vehicle."""
        world, units = self.ctx.world, self.book['units']
        plan = units['defenses']
        if world.spendable() is None:
            return False
        spendable = world.spendable() if spendable is None else spendable
        placed = sum(world.structures.get(i, 0) for i in plan['counted'])
        if (placed >= plan['max'] or not world.has('war_factory') or not world.has('ore_refinery')
                or world.units_built.get(units['tank'], 0) < plan['after_tanks']):
            return False
        if placed >= plan['proactive_count'] and not self.threatened(plan['after_threat_within'], now):
            return False
        if ((world.power.value or {}).get('level') == 'low'
                or world.lanes['defenses'].status in ('building', 'ready', 'hold')
                or any(c.status == 'requested' for c in world.ledger.open('queue', 'defenses'))):
            return False
        remaining = spendable - world.item(plan['item'])['cost']
        reserve = plan['reserve_tanks'] * world.item(units['tank'])['cost']
        miner = units['miners']
        _, miners, wanted = self.economy()
        if miners < wanted and not self.threatened(miner['save_unless_threat_within'], now):
            reserve = max(reserve, world.item(miner['item'])['cost'])
        return remaining >= reserve and not self.saving_for_refinery(remaining, now)

    def saving_for_refinery(self, spendable, now):
        """The factory waits while the second refinery is next and not yet paid for (not when
        enemies are at the base: then tanks come first)."""
        world, units = self.ctx.world, self.book['units']
        if world.structures.get('ore_refinery', 0) != 1 or not world.structures.get('war_factory'):
            return False
        if not self.tanks(units['second_refinery']['after_tanks'], None, now):
            return False
        if self.threatened(units['second_refinery']['save_unless_threat_within'], now):
            return False
        if getattr(world.lanes.get('buildings'), 'item', None) == 'ore_refinery':
            return False                          # already under construction
        return spendable < (world.item('ore_refinery')['cost'] +
                            units['second_refinery']['reserve_tanks'] * world.item(units['tank'])['cost'])

    def entry_due(self, entry, now, order):
        """Whether a build-order entry may be built now: its gate in the profile holds."""
        spendable = self.ctx.world.spendable() or 0
        if entry.get('high_tech') and not self.high_tech_ready(now):
            return False
        if 'with' in entry:                       # inherit the gate of the building it serves
            return self.entry_due(next(e for e in order if e['item'] == entry['with']), now, order)
        if spendable > self.book['buildings']['rich'] and not entry.get('even_when_rich'):
            return True                           # the factory cannot spend it fast enough anyway
        if 'tanks' in entry and not self.tanks(entry['tanks'], entry.get('or_after'), now):
            return False
        return 'credits' not in entry or spendable >= entry['credits']

    def next_building(self, available, now):
        world, plan = self.ctx.world, self.book['buildings']
        spendable = world.spendable()
        affordable = [i for i in available if spendable is None or
                      world.item(i)['cost'] <= max(spendable, 0) + plan['credit_slack']]

        def funded(item, serves=None):
            reserve = plan['tech_reserve']
            if ((item not in reserve['items'] and serves not in reserve['items']) or
                    not self.threatened(reserve['threat_within'], now)):
                return True
            tank_cost = world.item(self.book['units']['tank'])['cost']
            return (world.spendable() or 0) >= world.item(item)['cost'] + reserve['reserve_tanks'] * tank_cost

        power = (world.power.value or {}).get('level')
        # Replacing reactors under fire cannot restore a depleted army without a factory.
        if world.structures.get('war_factory') == 0 and 'war_factory' in affordable:
            return 'war_factory'
        # A just-placed reactor takes seconds to show on the power bar (live games 2 and 3 built
        # a second reactor at once because the bar still read low).
        if power == 'low' and now >= self.power_grace and plan['power'] in affordable:
            return plan['power']
        objective = self.intent.objective if self.intent else 'build_army'
        built = world.structures
        tech = plan['tech_up']
        if objective == 'tech_up' and self.high_tech_ready(now) and self.tanks(tech['tanks'], None, now):
            for item in tech['items']:
                if not built.get(item) and item in affordable:
                    return item if funded(item) else None
        counts = dict(built)
        order = plan['order']
        for entry in order:
            item = entry['item']
            if counts.get(item, 0) > 0:
                counts[item] -= 1
                continue
            if self.skip.get(item, 0) > now:
                continue                          # cancelled after failed placements: try later
            if not self.entry_due(entry, now, order):
                return None                       # its gate is closed: this and the rest of the order wait
            if not funded(item, entry.get('with')):
                return None
            if item in affordable:
                if self.awaited and self.awaited[0] == item:
                    self.awaited = None
                return item
            if item in available:
                self.awaited = None
                return None                       # save for a visible item instead of skipping it
            # Wait for the next item in order, but not forever (it may be locked or unreadable).
            if self.awaited is None or self.awaited[0] != item:
                self.awaited = (item, now)
            if now - self.awaited[1] < plan['wait_for_next']:
                return None
            # waited long enough: skip it and continue down the order
        after = plan['after_order']
        if built.get(after['item'], 0) < after['up_to'] and (world.spendable() or 0) > plan['rich'] \
                and after['item'] in affordable:
            return after['item']
        return None

    def next_item(self, lane, available, now):
        """Rule default, and a System One question when several sensible options exist."""
        world = self.ctx.world
        spendable = world.spendable()
        if lane == 'buildings':
            return self.next_building(available, now)
        options, needs = self.unit_options(lane, available, max(spendable or 0, 0) if spendable is not None else 10**6, now)
        if not options:
            return None
        default = options[0]
        if self.broker is None or len(options) < 2:
            return default
        chosen = self.choices.get(lane)
        if chosen and chosen[0] in options:
            return chosen[0]
        key = f'production:{lane}'
        if key not in self.broker.busy_keys():
            state = self.book.state('production', {'lane': lane, 'needs': needs or
                                                   [self.book.say('production', 'needs', 'nothing')]})
            self.broker.submit(Decision(key, 'production', state, self.book.question('production', options),
                                        self.broker.clock() + self.book.systemone['deadlines']['production'],
                                        self._decided(lane), {'production': {'type': 'choice', 'choice': default}},
                                        generation=self.generation()))
        return None          # decide next tick (answer or default arrives within the deadline)

    def _siege_limit(self):
        units, built = self.book['units'], self.ctx.world.units_built
        siege = units['siege']
        limit = siege['max']
        if siege.get('one_per_tanks'):
            limit = max(limit, self.tank_count() // siege['one_per_tanks'])
        if siege.get('one_per_heavy'):
            limit = max(limit, built.get(units['heavy_tank']['item'], 0) // siege['one_per_heavy'])
        return limit

    def queue_count(self, item):
        """How many to queue with one visit to the tab (two vehicles while money piles up)."""
        world, plan = self.ctx.world, self.book['units']['queue_two']
        spendable = world.spendable()
        lane = world.item(item)['lane']
        count = plan['count'] if (lane in plan['lanes'] and spendable is not None and
                                 spendable >= plan['credits_per_cost'] * world.item(item)['cost']) else plan['default_count']
        anti_air = self.book['units']['anti_air']
        if item == anti_air['item']:
            ratio = anti_air['one_per_tanks']
            wanted = (self.tank_count() + ratio - 1) // ratio
            count = min(count, max(1, wanted - world.units_built.get(item, 0)))
        if item == self.book['units']['miners']['item']:
            _, miners, wanted = self.economy()
            count = min(count, max(1, wanted - miners))
        siege = self.book['units']['siege']
        if item == siege['item']:
            limit = self._siege_limit()
            count = min(count, max(1, limit - world.units_built.get(item, 0)))
        return count

    # -- per tick ----------------------------------------------------------
    def step(self, snapshot, now=None, act=True):
        """Start at most one sidebar task. Returns a short description of the intended action."""
        now = now or self.clock()
        self.ask_strategy(now)
        plan = self.plan(snapshot, now)
        self.recommendation = plan
        if not act or plan is None or self.runner is None:
            return plan
        name, factory, lane = plan
        if self.runner.busy('army'):
            return plan  # Sidebar input would clear the selection being inspected by the army.
        if self.cooldown.get(name, 0) > now or self.runner.busy('sidebar'):
            return plan
        if self.runner.start(name, 'sidebar', factory(), priority=3):
            # Planning can run while another task owns the sidebar or in Recommend mode.
            # Consume System One's choice only once its production task is actually accepted.
            if name.startswith('queue:'):
                self.choices.pop(lane, None)
            self.cooldown[name] = now + 0.3
        return plan

    def ally_site(self, item, now):
        world, ctx = self.ctx.world, self.ctx
        plan = self.book.data['playbook'].get('team_play')
        radar = world.radar.value
        if (not plan or not ctx.build_off_ally or radar is None or not radar.online or
                world.radar.age(now) > 2 or ctx.radar_home is None or item not in plan['items'] or
                (self.intent is not None and self.intent.focus == 'attack') or
                world.structures.get('ore_refinery', 0) < plan['min_refineries'] or
                not world.has('war_factory') or (world.power.value or {}).get('level') != 'ok' or
                not any(world.has(i) for i in self.book['units']['defenses']['counted']) or
                self.threatened(plan['quiet_seconds'], now) or
                (world.spendable() or 0) < plan['reserve_tanks'] * world.item(self.book['units']['tank'])['cost']):
            return None
        bases = [p for area, p in radar.allies if area >= plan['min_base_dots'] and
                 not radar.enemy_near(p, plan['safe_radius'])]
        return min(bases, key=lambda p: distance(p, ctx.radar_home)) if bases else None

    def placement_target(self, item, failures, now):
        target = self.ally_site(item, now) if failures == 0 else None
        event = (item, target)
        if target is None:
            self.placement_event, self.placement_choice = event, 'home'
            return True, None
        if event != self.placement_event:
            self.placement_event, self.placement_choice = event, None
            default = self.book.rule('placement', ['home', 'ally_base'], {'site': 'ally_available'})
            if self.broker is None:
                self.placement_choice = default
            else:
                def decided(answers, provenance):
                    if event == self.placement_event:
                        self.placement_choice = answers['placement']['choice']
                        self.ctx.world.note('placement-choice', item=item, choice=self.placement_choice,
                                            target=list(target), source=provenance)
                state = self.book.state('placement', {'site': self.book.say('placement', 'site', 'ally_available')})
                self.broker.submit(Decision('placement', 'production', state,
                        self.book.question('placement', ['home', 'ally_base']),
                        self.broker.clock() + self.book.systemone['deadlines']['production'], decided,
                        {'placement': {'type': 'choice', 'choice': default}}, self.generation()))
        return self.placement_choice is not None, target if self.placement_choice == 'ally_base' else None

    def plan(self, snapshot, now):
        world, ctx = self.ctx.world, self.ctx
        lost = ctx.pointer is not None and getattr(ctx.pointer, 'name', '') == 'locked' and ctx.pointer.position is None
        deploying = snapshot.get('sidebar_empty') and not world.mcv_deployed.value
        # The MCV deploys by keys, so the first tick of a match deploys it; the pointer is
        # calibrated afterwards (or first, if the keys failed and the MCV must be clicked).
        if deploying and (not lost or not self.failures.get('deploy_mcv')):
            keys = not self.failures.get('deploy_mcv')
            return ('deploy_mcv', lambda: skills.deploy_mcv(ctx, keys=keys), 'buildings')
        if lost:
            return ('calibrate_pointer', lambda: skills.calibrate_pointer(ctx), 'pointer')
        lanes = snapshot['lanes']
        if lanes['buildings']['available'] and not world.structures.get('construction_yard'):
            world.structures['construction_yard'] = 1
            world.set('mcv_deployed', True)
        for lane in ('buildings', 'defenses'):
            view = lanes[lane]
            if view['status'] == 'ready' and view['item']:
                item = view['item']
                failures = self.failures.get(f'place:{item}', 0)
                if failures >= PLACE_ATTEMPTS:
                    # A finished building blocks all construction until placed: cancel it (refund).
                    return (f'cancel:{item}', lambda item=item: skills.cancel_ready(ctx, item), lane)
                ring = min(2, failures)
                decided, ally = self.placement_target(item, failures, now)
                if not decided:
                    return None
                if ally is not None:
                    return (f'place:{item}', lambda item=item, ally=ally:
                            skills.place_near_ally(ctx, item, ally, self.book['team_play']['max_probes']), lane)
                front = None
                if (lane == 'defenses' and ctx.camera == 'home' and snapshot['mode'] == 'hud' and
                        self.army is not None and self.army.seen is not None and
                        0 <= now - self.army.last_check <= self.army.every):
                    front = self.army.seen[0]
                elif lane == 'defenses' and ctx.radar_home is None:
                    dx, dy = self.book['units']['defenses']['front_without_radar']
                    front = (ctx.home[0] + dx, ctx.home[1] + dy)
                return (f'place:{item}', lambda item=item, ring=ring, front=front:
                        skills.place_building(ctx, item, ring, front=front), lane)
        posture = self.intent.posture if self.intent else 'gather'
        threatened = self.threatened(self.book['units']['defenses']['after_threat_within'], now)
        recon = self.reconnaissance_due()
        defenses = (posture == 'defend' or threatened or self.basic_defense_due(now) or self.fortification_due(now)
                    or recon
                    or lanes['defenses']['status'] in ('building', 'ready', 'hold'))
        unlocked = [lane for lane in ('buildings', 'vehicles', 'infantry', 'defenses')
                    if world.structures.get(LANE_UNLOCK[lane]) and lane not in world.disabled_lanes
                    and (lane != 'defenses' or defenses)]
        visible = snapshot['tab']
        view = lanes['defenses']
        if (recon and 'defenses' in unlocked and visible != 'defenses' and
                self.hold.get('defenses', 0) <= now and
                (view['age'] is None or view['age'] > self.lane_refresh or view['status'] == 'unknown')):
            # Learn the native recon cameo before another unit consumes its funding.
            return ('look:defenses', lambda: skills.select_tab(ctx, 'defenses'), 'defenses')
        # The visible lane first (no tab switch), then other lanes known to be idle, then stale ones.
        order = ([visible] if visible in unlocked else []) + [lane for lane in unlocked if lane != visible]
        for lane in order:
            view = lanes[lane]
            fresh = view['age'] is not None and view['age'] <= self.lane_refresh
            if fresh and view['status'] == 'idle' and self.hold.get(lane, 0) <= now:
                item = self.next_item(lane, view['available'], now)
                if item:
                    count = self.queue_count(item)
                    return (f'queue:{item}', lambda item=item, count=count: skills.queue_item(ctx, item, count), lane)
        stale = [lane for lane in order if lane != visible and self.hold.get(lane, 0) <= now and
                 (lanes[lane]['age'] is None or lanes[lane]['age'] > self.lane_refresh)]
        if stale:
            lane = max(stale, key=lambda l: lanes[l]['age'] if lanes[l]['age'] is not None else 1e9)
            return (f'look:{lane}', lambda lane=lane: skills.select_tab(ctx, lane), lane)
        return None

    def failed(self, task):
        """Runner failure hook: back off, and widen the search after failed placements."""
        self.failures[task.name] = self.failures.get(task.name, 0) + 1
        now = self.clock()
        # The MCV retries almost at once (the base cannot start without it); others back off, and a
        # cancel that does not work backs off more each time (live game 4 retried one 30 times).
        backoff = min(60.0, 3.0 * 2 ** (self.failures[task.name] - 1)) if task.name.startswith('cancel:') else 3.0
        self.cooldown[task.name] = now + (0.5 if task.name == 'deploy_mcv' else backoff)
        item = task.name.split(':', 1)[1] if task.name.startswith('queue:') else None
        lane = self.ctx.world.knowledge['items'].get(item, {}).get('lane') if item else None
        if task.name.startswith('look:') and self.failures[task.name] % 3 == 0:
            # The tab would not open: its building is probably gone (live game 1 retried 30 times).
            self.hold[task.name.split(':', 1)[1]] = now + 60.0
        if lane in UNIT_LANES and task.reason.startswith('expectation'):
            # The click most likely queued the unit even though the sidebar did not show it in
            # time: count it, and let the lane settle before queueing more (no runaway queues).
            world = self.ctx.world
            world.units_built[item] = world.units_built.get(item, 0) + 1
            self.hold[lane] = now + 8.0
        if self.log:
            self.log(dict(time=time.time(), failed=task.name, reason=task.reason,
                          failures=self.failures[task.name]))

    def done(self, task):
        """Runner completion hook: a cancelled building leaves the build order for a while;
        after a reactor is placed, low power is not trusted until the bar has caught up."""
        if task.name == f'place:{self.book["buildings"]["power"]}':
            self.power_grace = self.clock() + POWER_GRACE
        if task.name.startswith('cancel:'):
            item = task.name.split(':', 1)[1]
            self.skip[item] = self.clock() + SKIP_AFTER_CANCEL
            self.failures.pop(f'place:{item}', None)
