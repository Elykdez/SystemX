from pathlib import Path

import pytest
import cv2

from addons.chronodivide import playbook
from addons.chronodivide.army import Army, Campaign, attack_wave, combat_count, field_team, building_point
from addons.chronodivide.frame_player import FramePlayer
from addons.chronodivide.intent import StandingIntent
from addons.chronodivide.layout import Layout
from addons.chronodivide.miners import due, find_field
from addons.chronodivide.reflex import Reflex
from addons.chronodivide.skills import Context
from addons.chronodivide.world import Slot, World
from addons.chronodivide.client import ClientProfile
from addons.chronodivide.perception import colors
from addons.chronodivide.perception.radar import Radar, read_radar
from addons.chronodivide.perception.reader import Perception
from addons.chronodivide.perception.unitboxes import our_units
from addons.chronodivide.tests.test_unitboxes import draw_bar
from addons.chronodivide.tests.test_strategy import Recorder, run
from addons.chronodivide.tools.decisions import Capture
from addons.chronodivide.tasks import TaskRunner
from addons.chronodivide.executor import Trust
import numpy as np

PROFILE = Path(__file__).resolve().parents[1] / 'profiles' / 'startlux-ra2-allied.json'
FIXTURES = Path(__file__).parent / 'fixtures'


def game():
    book = playbook.load(PROFILE)
    world = World(knowledge=playbook.knowledge(book), clock=lambda: 500)
    ctx = Context(Layout.for_canvas(1024, 768), world)
    intent = StandingIntent()
    army = Army(ctx, None, intent, clock=lambda: 500, book=book)
    reflex = Reflex(ctx, intent=intent, clock=lambda: 500, army=army, book=book)
    world.structures.update(construction_yard=1, power_plant=3, ore_refinery=2,
                            barracks=1, war_factory=1, airforce_command=1, battle_lab=1)
    world.units_built.update(grizzly_tank=6, ifv=2, chrono_miner=2, attack_dog=2, gi=2)
    world.set('credits', 4000)
    world.set('power', {'level': 'ok'})
    return world, army, reflex


def test_frame_player_uses_selected_faction_knowledge_and_excludes_miners():
    player = FramePlayer(book=playbook.load(PROFILE))
    assert player.world.item('grizzly_tank')['cost'] == 700
    assert 'rhino_tank' not in player.world.knowledge['items']
    player.world.units_built.update(chrono_miner=4, grizzly_tank=2)
    assert combat_count(player.world, player.book) == 2


@pytest.mark.parametrize('tanks', [2, 4, 6, 7])
def test_opening_adds_a_miner_after_four_tanks_but_keeps_second_refinery_gated(tanks):
    world, army, reflex = game()
    world.structures.update(ore_refinery=1, airforce_command=0, battle_lab=0)
    world.units_built.update(grizzly_tank=tanks, chrono_miner=0, ifv=2)
    world.set('credits', 6000)
    assert reflex.next_building(['ore_refinery', 'airforce_command', 'power_plant'], 100) == (
        'airforce_command' if tanks >= 4 else None)
    options, _ = reflex.unit_options('vehicles', ['grizzly_tank', 'chrono_miner', 'ifv'], 6000, 100)
    assert options == (['chrono_miner', 'grizzly_tank'] if tanks >= 4 else ['grizzly_tank'])


def test_allied_opening_powers_radar_before_funding_the_second_refinery():
    world, _, reflex = game()
    world.structures.update(power_plant=1, ore_refinery=1, airforce_command=0, battle_lab=0)
    world.units_built.update(grizzly_tank=4, chrono_miner=0)
    available = ['power_plant', 'ore_refinery', 'airforce_command']
    assert reflex.next_building(available, 100) == 'power_plant'
    world.structures['power_plant'] = 2
    assert reflex.next_building(available, 100) == 'airforce_command'
    world.structures['airforce_command'] = 1
    assert reflex.next_building(available, 100) is None
    world.units_built['grizzly_tank'] = 5
    assert reflex.next_building(available, 100) is None
    world.units_built['grizzly_tank'] = 6
    assert reflex.next_building(available, 100) == 'ore_refinery'


def test_allied_second_miner_can_be_funded_before_saving_for_second_refinery():
    world, _, reflex = game()
    world.structures.update(ore_refinery=1, battle_lab=0)
    world.units_built.update(grizzly_tank=4, ifv=2, chrono_miner=0)
    world.set('credits', 1800)
    available = ['grizzly_tank', 'chrono_miner', 'ifv']
    options, _ = reflex.unit_options('vehicles', available, 1800, 500)
    assert options[0] == 'chrono_miner'
    world.units_built['chrono_miner'] = 1
    assert reflex.unit_options('vehicles', available, 1800, 500)[0] == ['grizzly_tank']
    world.units_built['grizzly_tank'] = 6
    assert reflex.unit_options('vehicles', available, 1800, 500)[0] == []


def test_surviving_factory_does_not_save_for_radar_without_a_construction_yard():
    world, army, reflex = game()
    world.structures.update(construction_yard=0, airforce_command=0)
    world.set('radar', Radar(False))
    options, _ = reflex.unit_options('vehicles', ['grizzly_tank', 'ifv'], 2000, 500)
    assert 'grizzly_tank' in options


def test_confirmed_factory_loss_rebuilds_production_before_more_power():
    world, _, reflex = game()
    world.structures['war_factory'] = 0
    world.set('power', {'level': 'low'})
    available = ['power_plant', 'war_factory']
    assert reflex.next_building(available, 500) == 'war_factory'
    world.structures['war_factory'] = 1
    assert reflex.next_building(available, 500) == 'power_plant'
    world.structures['war_factory'] = 0
    world.set('credits', 500)
    assert reflex.next_building(available, 500) is None  # neither building is funded
    assert reflex.next_building(['power_plant'], 500) is None


def test_native_allied_mcv_is_recognized_before_the_base_exists():
    from addons.chronodivide.calibration import find_game
    profile = ClientProfile.load('chronodivide-allied')
    canvas = cv2.imread(str(FIXTURES / 'allied-undeployed-native.png'))
    found = find_game(canvas, profile)
    assert found.rect.tuple() == (0, 0, 1024, 768)
    world = World(knowledge=playbook.knowledge(playbook.load(PROFILE)))
    perception = Perception(profile=profile)
    mode, readings, lane = perception.read(canvas, world)
    assert perception.modes.required == .7 and mode['mode'] == 'hud'
    assert world.credits.value == 10000 and world.sidebar_empty.value
    assert readings and lane is None and not world.structures


def test_nearly_finished_refinery_does_not_become_available_again():
    profile = ClientProfile.load('chronodivide-allied')
    catalog = profile.cameos()
    catalog.compressed = True
    crop = cv2.imread(str(FIXTURES / 'allied-refinery-nearly-ready.png'))
    reading = catalog.classify(crop)
    assert reading.item == 'ore_refinery' and reading.state == 'building'
    assert .95 < reading.progress < 1


def test_native_allied_1075_credit_readout_is_not_ambiguous_with_eight():
    profile = ClientProfile.load('chronodivide-allied')
    canvas = cv2.imread(str(FIXTURES / 'allied-credits-1075-native.png'))
    reader = profile.credits()
    reader.compressed = True
    assert reader.read(profile.layout().regions['credits'].crop(canvas)) == 1075


@pytest.mark.parametrize('quality', [None, 95, 90, 80])
def test_browser_allied_eight_is_not_read_as_zero(quality):
    reader = ClientProfile.load('chronodivide-allied').credits()
    reader.compressed = True
    crop = cv2.imread(str(FIXTURES / 'allied-credits-2800-browser.png'))
    if quality is not None:
        ok, encoded = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert reader.read(crop) == 2800


def test_native_radar_overlay_does_not_stop_allied_hud_at_seventy_percent():
    profile = ClientProfile.load('chronodivide-allied')
    reader = profile.modes(profile.layout())
    reader.compressed = True
    canvas = cv2.imread(str(FIXTURES / 'allied-hud-large-radar-overlay-native.png'))
    mode = reader.read(canvas)
    assert reader.required == .7 and mode['mode'] == 'hud'
    assert mode['scores']['top_buttons_upper'] >= .7
    assert mode['scores']['sidebar_bottom'] >= .7


def test_disabled_native_allied_production_hud_remains_playable_at_seventy_percent():
    profile = ClientProfile.load('chronodivide-allied')
    canvas = cv2.imread(str(FIXTURES / 'allied-disabled-production-native.png'))
    perception = Perception(profile=profile, compressed=True)
    world = World(knowledge=playbook.knowledge(playbook.load(PROFILE)))
    mode, readings, lane = perception.read(canvas, world)
    assert perception.modes.required == .7 and mode['mode'] == 'hud'
    assert mode['scores']['top_buttons'] >= .7 and mode['scores']['sidebar_bottom'] >= .7
    assert any(r.item == 'barracks' and r.state == 'ready' for r in readings)


def test_idle_native_allied_vehicle_icons_do_not_look_like_build_clocks():
    profile = ClientProfile.load('chronodivide-allied')
    catalog = profile.cameos()
    catalog.compressed = True
    canvas = cv2.imread(str(FIXTURES / 'allied-factory-present-native.png'))
    readings = [catalog.classify(box.crop(canvas)) for box in profile.layout().grid.boxes()[:2]]
    assert [(r.item, r.state) for r in readings] == [('chrono_miner', 'available'), ('grizzly_tank', 'available')]


def test_native_finished_grizzly_waiting_at_crowded_factory_exit_is_ready():
    profile = ClientProfile.load('chronodivide-allied')
    catalog = profile.cameos()
    catalog.compressed = True
    canvas = cv2.imread(str(FIXTURES / 'allied-factory-exit-blocked.png'))
    reading = catalog.classify(profile.layout().grid.box(1).crop(canvas))
    assert reading.item == 'grizzly_tank' and reading.state == 'ready' and reading.queued
    world = World(knowledge=playbook.knowledge(playbook.load(PROFILE)))
    _, _, lane = Perception(profile=profile, compressed=True).read(canvas, world)
    assert lane == 'vehicles'
    assert world.lanes[lane].status == 'ready' and world.lanes[lane].item == 'grizzly_tank'


def test_native_finished_prism_without_queue_numeral_still_blocks_the_factory():
    profile = ClientProfile.load('chronodivide-allied')
    canvas = cv2.imread(str(FIXTURES / 'allied-prism-ready-without-count.png'))
    world = World(knowledge=playbook.knowledge(playbook.load(PROFILE)))
    _, readings, lane = Perception(profile=profile, compressed=True).read(canvas, world)
    assert readings[3].item == 'prism_tank' and readings[3].state == 'ready'
    assert not readings[3].queued
    assert lane == 'vehicles'
    assert world.lanes[lane].status == 'ready' and world.lanes[lane].item == 'prism_tank'


def test_allied_tech_builds_mirage_screen_before_prisms_then_finishes_screen():
    world, army, reflex = game()
    world.structures['battle_lab'] = 1
    options, needs = reflex.unit_options('vehicles', ['grizzly_tank', 'ifv', 'mirage_tank', 'prism_tank'], 4000, 500)
    assert options == ['mirage_tank']
    world.units_built['mirage_tank'] = 1
    assert reflex.unit_options('vehicles', ['grizzly_tank', 'ifv', 'mirage_tank', 'prism_tank'], 4000, 500)[0] == ['mirage_tank']
    world.units_built['mirage_tank'] = 2
    options, needs = reflex.unit_options('vehicles', ['grizzly_tank', 'ifv', 'mirage_tank', 'prism_tank'], 4000, 500)
    assert options == ['prism_tank', 'grizzly_tank']
    assert 'long-range siege support' in needs
    world.units_built['prism_tank'] = 4
    options, _ = reflex.unit_options('vehicles', ['grizzly_tank', 'mirage_tank', 'prism_tank'], 4000, 500)
    assert options == ['mirage_tank']
    world.units_built['mirage_tank'] = 2
    options, _ = reflex.unit_options('vehicles', ['grizzly_tank', 'mirage_tank', 'prism_tank'], 4000, 500)
    assert options == ['mirage_tank']
    world.units_built['mirage_tank'] = 6
    options, _ = reflex.unit_options('vehicles', ['grizzly_tank', 'mirage_tank', 'prism_tank'], 4000, 500)
    assert options == ['mirage_tank', 'grizzly_tank']


def test_allied_mixed_wave_saves_for_screen_prisms_then_remaining_mirages():
    world, _, reflex = game()
    available = ['grizzly_tank', 'ifv', 'mirage_tank', 'prism_tank']
    assert reflex.unit_options('vehicles', available, 800, 500)[0] == []
    world.structures['battle_lab'] = 1
    world.units_built['mirage_tank'] = 2
    assert reflex.unit_options('vehicles', available, 1000, 500)[0] == []
    world.units_built['prism_tank'] = 4
    assert reflex.unit_options('vehicles', available, 700, 500)[0] == []


def test_allied_prisms_are_replaced_as_the_mirage_force_grows():
    world, _, reflex = game()
    world.units_built.update(prism_tank=4, mirage_tank=12)
    assert reflex.unit_options('vehicles', ['grizzly_tank', 'mirage_tank', 'prism_tank'], 1200, 500)[0] == ['prism_tank', 'grizzly_tank']
    assert reflex.unit_options('vehicles', ['grizzly_tank', 'mirage_tank', 'prism_tank'], 1000, 500)[0] == []


def test_allied_home_census_is_less_frequent_during_an_assault():
    _, army, _ = game()
    army.census_at = 470
    assert army._census_due(500)
    army.campaign.active = True
    assert not army._census_due(500)
    assert army._census_due(530)


def test_focus_persists_until_a_new_incident_and_ignores_a_late_answer():
    world, army, reflex = game()
    world.units_built['prism_tank'] = 1
    now = [500]
    reflex.clock = lambda: now[0]
    reflex.broker = Capture(reflex.clock)
    army.campaign.active, army.campaign.started = True, 480
    reflex.ask_focus(now[0])
    attack = reflex.broker.submitted[-1]
    assert reflex.intent.focus == 'attack'
    assert attack.deadline == now[0] + 8.0  # the immediate rule does not wait for a slow model reply
    attack.on_result(attack, {'focus': {'choice': 'attack'}}, 'systemone', {})
    assert reflex.intent.view()['provenance']['focus'] == 'systemone'
    for tick in (501, 504, 508):
        now[0] = tick
        reflex.ask_focus(tick)
    assert len(reflex.broker.submitted) == 1
    army.campaign.active = False
    army.threat_at = now[0]
    now[0] = 509
    reflex.ask_focus(now[0])
    assert len(reflex.broker.submitted) == 1       # the four-second visual check has not elapsed
    now[0] = 512
    army.threat_at = 512
    reflex.ask_focus(now[0])
    assert reflex.intent.focus == 'defend'
    assert len(reflex.broker.submitted) == 1       # emergency defense is the only feasible choice
    attack.on_result(attack, {'focus': {'choice': 'attack'}}, 'systemone', {})
    assert reflex.intent.focus == 'defend'
    assert reflex.intent.view()['provenance']['focus'] == 'rule'


@pytest.mark.parametrize('mirages,expected', [(0, 'mirage_tank'), (2, 'prism_tank')])
def test_allied_screen_and_prism_production_continue_during_a_recent_raid(mirages, expected):
    world, army, reflex = game()
    world.units_built['mirage_tank'] = mirages
    army.threat_at = 499
    options, _ = reflex.unit_options('vehicles', ['grizzly_tank', 'prism_tank', 'mirage_tank'], 1200, 500)
    assert expected in options
    assert ('mirage_tank' in options) == (mirages == 0)


def test_focus_changes_when_a_recent_defense_option_expires():
    _, army, reflex = game()
    now = [500]
    reflex.clock = lambda: now[0]
    reflex.broker = Capture(reflex.clock)
    army.threat_at = 480
    reflex.ask_focus(now[0])
    defense = reflex.broker.submitted[-1]
    defense.on_result(defense, {'focus': {'choice': 'defend'}}, 'systemone', {})
    assert reflex.intent.focus == reflex.intent.posture == 'defend'
    now[0] = 600
    reflex.ask_strategy(now[0])
    assert reflex.intent.focus == 'build' and reflex.intent.posture == 'gather'
    defense.on_result(defense, {'focus': {'choice': 'defend'}}, 'systemone', {})
    assert reflex.intent.focus == 'build'


@pytest.mark.parametrize('radar_contact', [False, True])
def test_our_red_building_preview_does_not_trigger_a_defense_incident(radar_contact):
    from addons.chronodivide.army import threat
    world, army, _ = game()
    previous = colors.use(colors.BLUE, (colors.RED,))
    try:
        world.set('mode', 'hud')
        army.ctx.camera = 'home'
        army.ctx.compressed_frames = True
        army.ctx.last_canvas = cv2.imread(str(FIXTURES / 'allied-placement-preview-browser.png'))
        assert threat(army.ctx.last_canvas, compressed=True)[1] > 1000
        army.runner = TaskRunner(Recorder(), Trust(), clock=lambda: 500)
        army.runner.start('place:power_plant', 'sidebar', iter(()))
        army.ctx.radar_home = (970, 110)
        world.set('radar', Radar(True, enemy=((4, (966, 110)),)) if radar_contact else Radar(False))
        army._look(500)
        assert army.seen is None
        assert (army.home_threat is not None) == radar_contact
        assert (army.threat_at == 500) == radar_contact
    finally:
        colors.use(*previous)


@pytest.mark.parametrize('profile', [PROFILE, playbook.PROFILE])
def test_both_factions_buy_two_extra_base_dogs_after_scouts_with_a_tank_reserved(profile):
    book = playbook.load(profile)
    world = World(knowledge=playbook.knowledge(book), clock=lambda: 500)
    ctx = Context(Layout.for_canvas(1024, 768), world)
    army = Army(ctx, book=book, clock=lambda: 500)
    army.scouts_ready = True
    reflex = Reflex(ctx, army=army, book=book, clock=lambda: 500)
    world.units_built['attack_dog'] = 2
    reserve = world.item(book['units']['tank'])['cost'] + world.item('attack_dog')['cost']
    assert reflex.unit_options('infantry', ['attack_dog'], reserve, 500)[0] == []
    world.structures['war_factory'] = 1
    assert reflex.unit_options('infantry', ['attack_dog'], reserve - 1, 500)[0] == []
    assert reflex.unit_options('infantry', ['attack_dog'], reserve, 500)[0] == ['attack_dog']
    world.units_built['attack_dog'] = 4
    assert reflex.unit_options('infantry', ['attack_dog'], reserve, 500)[0] == []
    assert book['attack']['keep_home_infantry']


@pytest.mark.parametrize('deselects', [True, False])
def test_field_team_preserves_home_infantry_and_requires_verified_deselection(deselects):
    world, army, _ = game()
    ctx = army.ctx
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    draw_bar(ctx.last_canvas, 200, 200, 18)
    draw_bar(ctx.last_canvas, 400, 200, 36)
    infantry = next(u for u in our_units(ctx.last_canvas) if u.kind == 'infantry')
    guard_point = (infantry.bar[0] + infantry.bar[2] // 2, infantry.bar[1] + 25)
    class SelectionGame(Recorder):
        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            if deselects and any(s.kind == 'click' and s.keys == ('shift',) and
                                 s.point == guard_point for s in steps):
                ctx.last_canvas[195:240, 180:235] = 0
    executor = SelectionGame()
    time = [500.0]
    runner = TaskRunner(executor, Trust(), clock=lambda: time[0])
    runner.start('field', 'army', field_team(ctx))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    keys = [s[2] for act in executor.acts for s in act[2]]
    assert (('ctrl', '1') in keys) == deselects
    if deselects:
        assert [u.kind for u in our_units(ctx.last_canvas)] == ['vehicle']


def test_field_team_uses_fresh_positions_when_home_infantry_moves():
    world, army, _ = game()
    ctx = army.ctx
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    for x in (200, 260):
        draw_bar(ctx.last_canvas, x, 200, 18)
    draw_bar(ctx.last_canvas, 400, 200, 36)
    clicked = []

    class MovingGuards(Recorder):
        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            for step in steps:
                if step.kind != 'click' or step.keys != ('shift',):
                    continue
                guards = [u for u in our_units(ctx.last_canvas) if u.kind == 'infantry']
                assert step.point in [(u.bar[0] + u.bar[2] // 2, u.bar[1] + 25) for u in guards]
                clicked.append(step.point)
                ctx.last_canvas[190:290, 180:340] = 0
                if len(clicked) == 1:
                    draw_bar(ctx.last_canvas, 300, 250, 18)

    time = [500.0]
    runner = TaskRunner(MovingGuards(), Trust(), clock=lambda: time[0])
    runner.start('field', 'army', field_team(ctx))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert len(clicked) == 2 and clicked[1][1] > clicked[0][1]
    assert [u.kind for u in our_units(ctx.last_canvas)] == ['vehicle']


def test_native_guard_ground_point_matches_successful_live_deselection():
    before = cv2.imread(str(FIXTURES / 'soviet-guard-selected-native.png'))
    after = cv2.imread(str(FIXTURES / 'soviet-guard-deselected-native.png'))
    guards = [u for u in our_units(before, compressed=True) if u.kind == 'infantry']
    assert len(guards) == 1
    guard = guards[0]
    point = (guard.bar[0] + guard.bar[2] // 2, guard.bar[1] + 25)
    # The live successful click was at canvas (203, 451); crop origin is (180, 420).
    assert abs(point[0] - 23) <= 1 and point[1] == 31
    assert guard.center[1] < point[1] - 5
    assert not [u for u in our_units(after, compressed=True) if u.kind == 'infantry']


@pytest.mark.parametrize('vehicles', [False, True])
def test_failed_team_formation_does_not_start_wave_or_guess_timeout(vehicles):
    world, army, _ = game()
    ctx = army.ctx
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    draw_bar(ctx.last_canvas, 200, 200, 18)
    if vehicles:
        draw_bar(ctx.last_canvas, 400, 200, 36)
    campaign = Campaign()
    campaign.active, campaign.kind = True, 'guess'
    time = [500.0]
    runner = TaskRunner(Recorder(), Trust(), clock=lambda: time[0])
    runner.start('wave', 'army', attack_wave(ctx, (940, 90), 'guess', campaign=campaign,
                                          keep_home_infantry=True))
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert campaign.waves == 0 and campaign.guess_since is None
    assert not campaign.team_formed and campaign.last_wave == -1e9


def test_guess_timeout_starts_after_delayed_attack_order():
    world, army, _ = game()
    time = [500.0]
    world.clock = lambda: time[0]
    campaign = Campaign()
    campaign.active, campaign.kind = True, 'guess'
    runner = TaskRunner(Recorder(), Trust(), clock=lambda: time[0])
    runner.start('wave', 'army', attack_wave(army.ctx, (940, 90), 'guess', campaign=campaign))
    runner.tick(world.snapshot(), None)
    time[0] += 100
    assert campaign.waves == 0 and campaign.guess_since is None
    assert run(runner, world, time, lambda: not runner.busy('army'))
    assert campaign.waves == 1 and campaign.guess_since >= 600
    assert campaign.last_wave == campaign.guess_since


@pytest.mark.parametrize('name,found', [('ore-rich-browser.png', True), ('ore-depleted-browser.png', False)])
def test_allied_resource_survey_can_find_coherent_ore_as_well_as_gems(name, found):
    canvas = np.zeros((768, 1024, 3), np.uint8)
    canvas[600:736, 500:850] = cv2.imread(str(FIXTURES / name))
    plan = playbook.load(PROFILE)['resources']
    result = find_field(canvas, (0, 0, 850, 736), plan, (425, 368))
    assert (result is not None) == found
    if found:
        assert result[2] == 'ore' and result[1] >= 12
    assert find_field(canvas, (0, 0, 850, 736), playbook.load()['resources'], (425, 368)) is None


@pytest.mark.parametrize('block', [None, 'power', 'radar', 'partial', 'gate', 'anchors', 'one_lane'])
def test_lost_battle_lab_needs_powered_radar_and_two_complete_missing_unlock_lanes(block):
    world, _, _ = game()
    plan = world.knowledge['technology_loss']
    if block == 'power':
        world.set('power', {'level': 'low'})
    for now in (500, 501, 502):
        world.set('radar', Radar(block != 'radar'), now)
        for lane, anchors in plan['anchors'].items():
            if block == 'one_lane' and lane == 'vehicles':
                continue
            items = list(anchors)
            if block == 'gate':
                items += plan['gates'][lane]
            if block == 'anchors':
                items[0] = None
            slots = [Slot(i, item, 'available') for i, item in enumerate(items)] + [Slot(len(items), None, 'empty')]
            world.reconcile_structures(lane, slots, complete=block != 'partial', now=now)
    assert world.structures['battle_lab'] == (0 if block is None else 1)
    if block is None:
        assert any(e.get('item') == 'battle_lab' and e['kind'] == 'structure-lost' for e in world.events)


@pytest.mark.parametrize('block', [None, 'power', 'radar', 'partial', 'gate', 'anchors', 'one_lane'])
def test_allied_radar_loss_requires_power_and_two_complete_missing_technology_lanes(block):
    world, _, _ = game()
    world.set('radar', Radar(block == 'radar'), 500)
    if block == 'power':
        world.set('power', {'level': 'low'})
    plan = world.knowledge['radar_loss']
    for now in (500, 501, 502):
        world.set('radar', Radar(block == 'radar'), now)
        for lane, anchors in plan['anchors'].items():
            if block == 'one_lane' and lane == 'vehicles':
                continue
            items = list(anchors)
            if block == 'gate':
                items += plan['gates'][lane]
            if block == 'anchors':
                items[0] = None
            slots = [Slot(i, item, 'available') for i, item in enumerate(items)] + [Slot(len(items), None, 'empty')]
            world.reconcile_structures(lane, slots, complete=block != 'partial', now=now)
    assert world.structures['airforce_command'] == (0 if block is None else 1)


def test_recorded_lost_allied_radar_has_power_and_no_rocketeer_option():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        world = World(knowledge=playbook.knowledge(playbook.load(PROFILE)))
        frame = cv2.imread(str(FIXTURES / 'allied-radar-lost-browser.png'))
        _, slots, lane = Perception(profile=profile, compressed=True).read(frame, world, now=500)
        assert lane == 'infantry'
        assert world.power.value['level'] == 'ok' and not world.radar.value.online
        assert [s.item for s in slots[:3]] == ['gi', 'engineer', 'attack_dog']
        assert 'rocketeer' not in [s.item for s in slots]
    finally:
        colors.use(*previous)


def test_recorded_missing_harrier_and_rocketeer_confirm_radar_loss_even_with_nighthawk_available():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        world, _, _ = game()
        perception = Perception(profile=profile, compressed=True)
        for name in ('allied-radar-lost-browser.png', 'allied-radar-lost-vehicles-browser.png'):
            frame = cv2.imread(str(FIXTURES / name))
            for now in (500, 501, 502) if 'vehicles' not in name else (503, 504, 505):
                _, slots, _ = perception.read(frame, world, now=now)
        assert 'nighthawk' in [s.item for s in slots]
        assert world.structures['airforce_command'] == 0
    finally:
        colors.use(*previous)


def test_building_click_uses_a_colored_surface_when_the_cluster_center_is_empty():
    previous = colors.use(colors.HouseColor('blue'), (colors.HouseColor('red'),))
    try:
        canvas = np.zeros((768, 1024, 3), np.uint8)
        canvas[300:320, 400:420] = (20, 20, 220)
        canvas[305:315, 405:415] = 0
        point = building_point(canvas, (300, (410, 310), (400, 300, 20, 20)))
        assert point != (410, 310)
        assert point[1] == 319
        assert colors.enemy_mask(canvas)[point[1], point[0]]
    finally:
        colors.use(*previous)


def test_unarmed_chrono_miner_cannot_be_used_as_emergency_defender():
    _, army, _ = game()
    army.miners.grouped = True
    assert not due(army, 500)


@pytest.mark.parametrize('age,cash,expected', [
    (0, 700, ['grizzly_tank']),
    (20, 700, ['grizzly_tank']),
    (21, 700, []),
    (0, 699, []),
])
def test_radar_recovery_can_fund_a_basic_defender_during_a_fresh_raid(age, cash, expected):
    world, army, reflex = game()
    world.structures['airforce_command'] = 0
    world.set('radar', Radar(False), 500)
    army.threat_at = 500 - age
    options, _ = reflex.unit_options('vehicles', ['grizzly_tank', 'ifv', 'mirage_tank'], cash, 500)
    assert options == expected


@pytest.mark.parametrize('missing', ['radar', 'funds', 'power', 'tanks', 'quota'])
def test_grand_cannon_requires_funded_powered_mobile_defense(missing):
    world, _, reflex = game()
    world.units_built['mirage_tank'] = 2
    world.structures['pillbox'] = 1
    if missing == 'radar':
        world.structures['airforce_command'] = 0
    elif missing == 'funds':
        world.set('credits', 3399)
    elif missing == 'power':
        world.set('power', {'level': 'low'})
    elif missing == 'tanks':
        world.units_built['grizzly_tank'] = 0
    else:
        world.structures['grand_cannon'] = 1
    options, _ = reflex.unit_options('defenses', ['grand_cannon'], world.spendable(), 500)
    assert options == []


def test_grand_cannon_is_offered_with_mobile_production_reserve():
    world, _, reflex = game()
    world.units_built['mirage_tank'] = 2
    world.structures['pillbox'] = 1
    world.structures['battle_lab'] = 0
    options, _ = reflex.unit_options('defenses', ['grand_cannon'], 3400, 500)
    assert options == ['grand_cannon']
    assert reflex.unit_options('defenses', ['prism_tower'], 3400, 500)[0] == []


@pytest.mark.parametrize('mirages', [0, 1, 2])
def test_optional_cannon_waits_for_the_first_mirage_screen(mirages):
    world, _, reflex = game()
    world.structures['pillbox'] = 1
    world.units_built['mirage_tank'] = mirages
    assert reflex.unit_options('defenses', ['grand_cannon'], 4000, 500)[0] == (
        ['grand_cannon'] if mirages >= 2 else [])


def test_native_grand_cannon_cameo_available_before_battle_lab():
    profile = ClientProfile.load('chronodivide-allied')
    catalog = profile.cameos()
    catalog.compressed = True
    canvas = cv2.imread(str(FIXTURES / 'allied-grand-cannon-without-lab.png'))
    reading = catalog.classify(profile.layout().grid.box(4).crop(canvas))
    assert reading.item == 'grand_cannon' and reading.state == 'available'


def test_allied_refinery_loss_uses_allied_sidebar_anchors():
    world, _, _ = game()
    unlocked = [Slot(i, item, 'available') for i, item in enumerate(
        ['power_plant', 'ore_refinery', 'barracks', 'war_factory', 'airforce_command'])] + [Slot(5, None, 'empty')]
    world.reconcile_structures('buildings', unlocked, complete=True, now=500)
    missing = unlocked[:3] + [Slot(3, None, 'empty')]
    for now in [501, 502, 503]:
        world.reconcile_structures('buildings', missing, complete=True, now=now)
    assert world.structures['ore_refinery'] == 0


def test_allied_tech_budget_holds_basic_tanks_until_lab_is_built():
    world, army, reflex = game()
    world.structures['battle_lab'] = 0
    world.set('credits', 1500)
    army.census_at = 500
    army.census_tanks = army.census_healthy_vehicles = 8
    assert not reflex.high_tech_ready(500)
    assert reflex.high_tech_ready(500, funded=False)
    assert reflex.unit_options('vehicles', ['grizzly_tank', 'ifv', 'chrono_miner'], 1500, 500)[0] == []
    world.set('credits', 3000)
    assert reflex.high_tech_ready(500)
    world.structures['battle_lab'] = 1
    assert reflex.unit_options('vehicles', ['grizzly_tank', 'ifv'], 1500, 500)[0] == ['grizzly_tank']


def test_allied_vehicle_production_reserves_funds_to_rebuild_lost_radar():
    world, _, reflex = game()
    world.structures['airforce_command'] = 0
    world.set('radar', Radar(False))
    assert reflex.unit_options('vehicles', ['grizzly_tank', 'mirage_tank', 'prism_tank'], 1500, 500)[0] == []


def test_recorded_allied_hud_reads_white_radar_blue_credits_and_available_infantry():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        world = World(knowledge=playbook.knowledge(playbook.load(PROFILE)))
        frame = cv2.imread(str(FIXTURES / 'allied-radar-browser.png'))
        mode, slots, lane = Perception(profile=profile, compressed=True).read(frame, world)
        assert mode['mode'] == 'hud'
        assert world.credits.value == 250
        assert lane == 'infantry'
        assert [s.item for s in slots[:4]] == ['gi', 'engineer', 'attack_dog', 'rocketeer']
        assert all(s.state == 'available' and not s.queued for s in slots[:4])
        radar = world.radar.value
        assert radar.online
        assert radar.bounds == (886, 48, 991, 157)
        assert radar.view == (961, 92, 978, 120)
        assert radar.own[0][1] == (971, 111)
        assert radar.enemy[0][1] == (932, 131)
    finally:
        colors.use(*previous)


def test_allied_radar_rejects_offline_faction_emblem():
    frame = cv2.imread(str(FIXTURES / 'allied-base-browser.png'))
    profile = ClientProfile.load('chronodivide-allied')
    assert not read_radar(frame, profile.layout().regions['radar'], True, line_color='white').online


def test_native_allied_radar_reads_pale_blue_map_and_camera_borders():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        frame = cv2.imread(str(FIXTURES / 'allied-factory-lost-native.png'))
        radar = read_radar(frame, profile.layout().regions['radar'], line_color='white')
        assert radar.online and radar.bounds == (886, 48, 991, 157)
        assert radar.view is not None and radar.own
        assert radar.in_view(radar.own[0][1])
    finally:
        colors.use(*previous)


def test_native_aircraft_only_sidebar_confirms_factory_loss_and_requests_rebuild():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        world, _, reflex = game()
        perception = Perception(profile=profile)
        frame = cv2.imread(str(FIXTURES / 'allied-factory-lost-native.png'))
        for now in (500, 501, 502):
            _, slots, lane = perception.read(frame, world, now=now)
        assert lane == 'vehicles' and slots[0].item == 'harrier'
        assert world.structures['war_factory'] == 0
        assert world.structures['airforce_command'] == 1
        assert 'vehicles' not in world.disabled_lanes
        assert world.lanes['vehicles'].available == ('harrier',)
        world.update_sidebar('buildings', [Slot(0, 'war_factory', 'available')], 502)
        assert reflex.plan(world.snapshot(), 502)[0] == 'queue:war_factory'
    finally:
        colors.use(*previous)


@pytest.mark.parametrize('block', ['ground', 'partial', 'unknown', 'empty', 'scrolled', 'recent', 'gap', 'same_time'])
def test_factory_loss_requires_repeated_complete_aircraft_only_top_of_list(block):
    world, _, _ = game()
    slots = [Slot(0, 'harrier', 'available'), Slot(1, None, 'empty')]
    if block == 'ground':
        slots.insert(1, Slot(1, 'grizzly_tank', 'disabled'))
    elif block == 'unknown':
        slots.insert(1, Slot(1, None, 'unknown'))
    elif block == 'empty':
        slots = [Slot(0, None, 'empty')]
    elif block == 'scrolled':
        slots = [slots[0]]
    elif block == 'recent':
        world._last_placement = 500
    times = (500, 511, 522) if block == 'gap' else (500, 500, 500) if block == 'same_time' else (500, 501, 502)
    for now in times:
        world.reconcile_structures('vehicles', slots, complete=block != 'partial', now=now)
    assert world.has('war_factory')


def test_factory_loss_preserves_aircraft_orders_and_clears_destroyed_ground_queue():
    world, _, _ = game()
    ground = world.ledger.add('queue', 'vehicles', 'grizzly_tank', cost=700)
    aircraft = world.ledger.add('queue', 'vehicles', 'harrier', cost=1200)
    slots = [Slot(0, 'harrier', 'building', .5), Slot(1, None, 'empty')]
    for now in (500, 501, 502):
        world.reconcile_structures('vehicles', slots, complete=True, now=now)
    assert not world.has('war_factory')
    assert ground.status == 'cancelled' and aircraft.status == 'requested'
    assert world.ledger.reserved() == 1200


def test_native_ground_vehicle_sidebar_preserves_factory():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        world, _, _ = game()
        perception = Perception(profile=profile)
        frame = cv2.imread(str(FIXTURES / 'allied-factory-present-native.png'))
        for now in (500, 501, 502):
            _, slots, lane = perception.read(frame, world, now=now)
        assert lane == 'vehicles' and slots[0].item == 'chrono_miner'
        assert world.has('war_factory')
    finally:
        colors.use(*previous)


def test_sandbox_player_emits_factory_rebuild_from_recorded_loss_and_building_frames():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        now = [500.0]
        player = FramePlayer(profile=profile, book=playbook.load(PROFILE), clock=lambda: now[0])
        player.world.structures.update(construction_yard=1, power_plant=2, ore_refinery=2,
                                       barracks=1, war_factory=1, airforce_command=1)
        frame = cv2.imread(str(FIXTURES / 'allied-factory-lost-native.png'))
        for t in (500, 501, 502):
            now[0] = t
            player.perception.read(frame, player.world, now=t)
        assert not player.world.has('war_factory')
        frame[:, 850:] = cv2.imread(str(FIXTURES / 'allied-factory-rebuild-sidebar-native.png'))
        result = player.step(frame)
        action = result['transaction']
        assert action['task'] == 'queue:war_factory'
        assert any(s['kind'] == 'click' and tuple(s['point']) == (971, 302) for s in action['steps'])
        assert result['awaiting'] == action['id']
        assert not player.world.has('war_factory')  # a queued command cannot claim a successful rebuild
    finally:
        colors.use(*previous)


def test_allied_activation_flash_does_not_create_a_false_radar_home():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        player = FramePlayer(profile=profile, book=playbook.load(PROFILE), compressed=True, clock=lambda: 500)
        canvas = cv2.imread(str(FIXTURES / 'allied-radar-activation-flash-native.png'))
        result = player.step(canvas)
        assert result['state'] == 'hud'
        assert not player.world.radar.value.online
        assert player.ctx.radar_home is None
        canvas = cv2.imread(str(FIXTURES / 'allied-radar-after-activation-native.png'))
        player.step(canvas, result['awaiting'])
        assert player.world.radar.value.online
        assert player.ctx.radar_home == (971, 111)
    finally:
        colors.use(*previous)


def test_native_offscreen_raider_reaches_defense_policy_through_radar():
    from addons.chronodivide.army import threat
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        world, army, reflex = game()
        frame = cv2.imread(str(FIXTURES / 'allied-offscreen-raider-native.png'))
        assert threat(frame) is None
        Perception(profile=profile).read(frame, world, now=500)
        radar = world.radar.value
        assert radar.online and any(not radar.in_view(p, margin=0) for _, p in radar.enemy)
        army.ctx.last_canvas, army.ctx.radar_home = frame, radar.own[0][1]
        army.census_enabled, army.scouts_ready = False, True
        army._look(500)
        assert army.seen is None and army.home_threat is not None
        assert reflex.feasible_postures(500) == ['defend']
        army.intent.posture = 'defend'
        order = army.step(world.snapshot(), now=500, act=False)
        assert order == ('army:defend-radar', army.home_threat[0])
    finally:
        colors.use(*previous)


@pytest.mark.parametrize('block', [None, 'known_enemy', 'no_lab', 'low_power', 'unfunded', 'already_built', 'missing_cameo'])
def test_allied_reconnaissance_requires_native_availability_and_funds(block):
    from addons.chronodivide.intel import Intel
    world, army, reflex = game()
    world.units_built['mirage_tank'] = 2
    army.intel = Intel(lambda: 500, book=army.book)
    available, credits = ['spy_satellite'], 2200
    if block == 'known_enemy':
        army.intel.enemy_base = (900, 100)
    elif block == 'no_lab':
        world.structures['battle_lab'] = 0
    elif block == 'low_power':
        world.set('power', {'level': 'low'})
    elif block == 'unfunded':
        credits -= 1
    elif block == 'already_built':
        world.structures['spy_satellite'] = 1
    elif block == 'missing_cameo':
        available = []
    options, _ = reflex.unit_options('defenses', available, credits, 500)
    assert ('spy_satellite' in options) == (block is None)


def test_unknown_allied_enemy_reserves_native_reconnaissance_but_a_fresh_raid_keeps_defenders():
    from addons.chronodivide.intel import Intel
    world, army, reflex = game()
    world.units_built['mirage_tank'] = 2
    army.intel = Intel(lambda: 500, book=army.book)
    world.lanes['defenses'].available = ('spy_satellite',)
    available = ['grizzly_tank', 'mirage_tank', 'prism_tank', 'ifv']
    assert reflex.unit_options('vehicles', available, 2100, 500)[0] == []
    army.threat_at = 500
    assert reflex.unit_options('vehicles', available, 2100, 500)[0]
    army.threat_at = -1e9
    world.lanes['defenses'].available = ()
    assert reflex.unit_options('vehicles', available, 2100, 500)[0]


def test_native_reconnaissance_funding_is_not_spent_on_optional_fortification():
    from addons.chronodivide.intel import Intel
    world, army, reflex = game()
    world.units_built['mirage_tank'] = 2
    army.intel = Intel(lambda: 500, book=army.book)
    available = ['spy_satellite', 'grand_cannon']
    assert reflex.unit_options('defenses', available, 4000, 500)[0] == ['spy_satellite']
    army.intel.enemy_base = (900, 100)
    assert reflex.unit_options('defenses', available, 4000, 500)[0] == ['grand_cannon']


def test_quiet_allied_scheduler_inspects_and_funds_native_reconnaissance():
    from addons.chronodivide.intel import Intel
    world, army, reflex = game()
    world.units_built['mirage_tank'] = 2
    army.intel = Intel(lambda: 500, book=army.book)
    world.structures['pillbox'] = 1
    world.set('credits', 2200)
    world.update_sidebar('vehicles', [Slot(1, 'grizzly_tank', 'available')], now=500)
    world.update_sidebar('buildings', [], now=500)
    world.update_sidebar('infantry', [], now=500)
    world.active_tab.value = 'vehicles'
    action = reflex.plan(world.snapshot(), 500)
    assert action[0] == 'look:defenses'
    world.update_sidebar('defenses', [Slot(6, 'spy_satellite', 'available')], now=500)
    action = reflex.plan(world.snapshot(), 500)
    assert action[0] == 'queue:spy_satellite'


@pytest.mark.parametrize('mirages', [0, 1])
def test_optional_reconnaissance_cannot_block_the_first_mirage_screen(mirages):
    from addons.chronodivide.intel import Intel
    world, army, reflex = game()
    army.intel = Intel(lambda: 500, book=army.book)
    world.units_built['mirage_tank'] = mirages
    world.lanes['defenses'].available = ('spy_satellite',)
    assert not reflex.reconnaissance_due(['spy_satellite'])
    assert reflex.unit_options('vehicles', ['mirage_tank', 'prism_tank', 'grizzly_tank'], 2200, 500)[0] == ['mirage_tank']
    assert 'spy_satellite' not in reflex.unit_options('defenses', ['spy_satellite'], 2200, 500)[0]


def test_allied_lab_defense_sidebar_includes_spy_satellite_in_the_defense_lane():
    profile = ClientProfile.load('chronodivide-allied')
    previous = profile.use_colors()
    try:
        world = World(knowledge=playbook.knowledge(playbook.load(PROFILE)))
        frame = cv2.imread(str(FIXTURES / 'allied-lab-defenses-browser.png'))
        _, slots, lane = Perception(profile=profile, compressed=True).read(frame, world)
        assert lane == 'defenses'
        assert [s.item for s in slots[:7]] == ['allied_wall', 'pillbox', 'patriot', 'prism_tower',
                                             'gap_generator', 'grand_cannon', 'spy_satellite']
        assert all(s.state == 'available' for s in slots[:7])
    finally:
        colors.use(*previous)


def test_allied_attack_moves_on_from_a_still_shrouded_guess_without_claiming_a_loss():
    world, army, _ = game()
    home = (970, 111)
    radar = Radar(True, (886, 48, 991, 157), own=((60, home),))
    world.set('radar', radar)
    army.ctx.radar_home = home
    first, second = radar.guesses(home)[:2]
    campaign = army.campaign
    campaign.active, campaign.teams = True, False
    campaign.started = campaign.guess_since = 400
    campaign.last_wave, campaign.target, campaign.kind = 400, first, 'guess'
    army.census_enabled = False
    assert army._attack(500, False) == ('army:attack', second)
    assert campaign.cleared == [first]
    assert army.failures == 0
