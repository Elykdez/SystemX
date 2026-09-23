"""Team relationships, multiple targets and verified ally-side construction."""

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from addons.chronodivide import playbook, skills
from addons.chronodivide.army import choose_target, threat
from addons.chronodivide.client import ClientProfile
from addons.chronodivide.perception import colors
from addons.chronodivide.perception.radar import Radar, read_radar
from addons.chronodivide.tasks import Act, Look, Poll, Wait
from addons.chronodivide.tests.test_strategy import LAYOUT, setup
from addons.chronodivide.world import Slot


GREEN = colors.HouseColor.sample('green', np.full((8, 8, 3), (40, 200, 40), np.uint8))
PURPLE = colors.HouseColor.sample('purple', np.full((8, 8, 3), (180, 40, 180), np.uint8))


@pytest.fixture
def team_colors():
    previous = colors.use(colors.BLUE, (colors.RED, PURPLE), (GREEN,))
    yield
    colors.use(*previous)


def test_team_profile_round_trip_and_color_conflicts(tmp_path):
    profile = ClientProfile.load('chronodivide-allied').derived('Team match')
    profile.path = tmp_path / 'team.json'
    profile.set_colors(enemies=(colors.RED, PURPLE), allies=(GREEN,))
    profile.raw['team_play'] = {'build_off_ally': True}
    profile.save()
    restored = ClientProfile.load(profile.path)
    assert restored.enemies == (colors.RED, PURPLE) and restored.allies == (GREEN,)
    assert restored.raw['team_play']['build_off_ally']
    with pytest.raises(ValueError, match='exactly one side'):
        restored.set_colors(allies=(colors.RED,))
    assert restored.allies == (GREEN,)


@pytest.mark.parametrize('compressed', [False, True])
def test_two_enemies_and_ally_are_separate_on_radar_and_tactical_view(team_colors, compressed):
    canvas = np.full((768, 1024, 3), 100, np.uint8)
    canvas[250:275, 300:330] = (40, 200, 40)
    assert threat(canvas, compressed=compressed) is None
    canvas[250:275, 550:580] = (24, 24, 255)
    canvas[350:375, 600:630] = (180, 40, 180)
    mask = colors.enemy_mask(canvas, compressed)
    assert not mask[250:275, 300:330].any()
    assert mask[250:275, 550:580].any() and mask[350:375, 600:630].any()
    box = LAYOUT.regions['radar']
    crop = box.crop(canvas)
    cv2.rectangle(crop, (8, 5), (135, 105), (0, 255, 255), 1)
    crop[65:69, 20:24] = (212, 105, 34)
    crop[65:71, 55:61] = (40, 200, 40)
    crop[25:30, 55:60] = (24, 24, 255)
    crop[25:30, 100:105] = (180, 40, 180)
    if compressed:
        canvas = cv2.imdecode(cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], 1)
    radar = read_radar(canvas, box, compressed)
    assert radar.online and len(radar.enemy) == 2 and len(radar.allies) == 1
    ally = radar.allies[0][1]
    assert not radar.enemy_near(ally, 10)
    assert radar.summary()['allies']


def test_one_against_three_finishes_each_hostile_base():
    home = (100, 100)
    first, second, third = (200, 100), (280, 100), (350, 200)
    radar = Radar(True, (0, 0, 400, 250), enemy=((16, first), (120, second), (50, third)),
                  allies=((200, (200, 200)),))
    assert choose_target(radar, home, current=first) == (first, 'enemy')
    radar = replace(radar, enemy=((120, second), (50, third)))
    assert choose_target(radar, home, current=first, known=first) == (second, 'enemy')
    assert choose_target(replace(radar, enemy=((50, third),)), home, known=second) == (third, 'enemy')


def test_hidden_search_does_not_choose_revealed_ally_base():
    radar = Radar(True, (0, 0, 100, 100), allies=((40, (80, 80)),))
    assert (80, 80) not in radar.guesses((20, 20))
    assert choose_target(radar, (20, 20), current=(80, 80)) != ((80, 80), 'guess')


def team_game():
    world, ctx, runner, army, reflex, executor, time = setup()
    ctx.build_off_ally = True
    ctx.radar_home = (900, 100)
    world.structures.update(ore_refinery=2, war_factory=1, sentry_gun=1, tesla_reactor=3)
    world.set('credits', 4000)
    world.set('power', {'level': 'ok'})
    world.set('radar', Radar(True, (880, 50, 1000, 150), allies=((20, (945, 100)),)))
    return world, ctx, army, reflex, time


@pytest.mark.parametrize('blocked', ['none', 'disabled', 'attack', 'raid', 'low_power', 'poor', 'opening', 'enemy', 'stale', 'item'])
def test_ally_construction_yields_to_home_and_combat(blocked):
    world, ctx, army, reflex, time = team_game()
    item = 'tesla_reactor'
    if blocked == 'disabled': ctx.build_off_ally = False
    elif blocked == 'attack': reflex.intent.focus = 'attack'
    elif blocked == 'raid': army.threat_at = time[0]
    elif blocked == 'low_power': world.set('power', {'level': 'low'})
    elif blocked == 'poor': world.set('credits', 1000)
    elif blocked == 'opening': world.structures['ore_refinery'] = 1
    elif blocked == 'enemy': world.set('radar', replace(world.radar.value, enemy=((4, (943, 100)),)))
    elif blocked == 'stale': time[0] += 3
    elif blocked == 'item': item = 'ore_refinery'
    assert reflex.ally_site(item, time[0]) == ((945, 100) if blocked == 'none' else None)


def test_placement_decision_discards_late_reply_after_home_becomes_threatened():
    from addons.chronodivide.tools.decisions import Capture
    world, ctx, army, reflex, time = team_game()
    reflex.broker = Capture(lambda: time[0])
    assert reflex.placement_target('tesla_reactor', 0, time[0]) == (False, None)
    question = reflex.broker.submitted[-1]
    army.threat_at = time[0]
    assert reflex.placement_target('tesla_reactor', 0, time[0]) == (True, None)
    question.on_result({'placement': {'choice': 'ally_base'}}, 'late')
    assert reflex.placement_choice == 'home'


@pytest.mark.parametrize('valid', [False, True])
def test_remote_placement_uses_green_footprint_or_returns_home(monkeypatch, team_colors, valid):
    world, ctx, army, reflex, time = team_game()
    ctx.last_canvas = np.full((768, 1024, 3), 100, np.uint8)
    ctx.last_canvas[300:340, 400:440] = (40, 200, 40)
    world.update_sidebar('buildings', [Slot(0, 'tesla_reactor', 'ready')])
    monkeypatch.setattr(skills, 'read_ghost', lambda *args: SimpleNamespace(seen=True,
            valid=valid or ctx.camera == 'home', red_center=None, green_center=None))
    def home(ctx):
        ctx.camera = 'home'
        yield Act([])
    monkeypatch.setattr(skills, 'center_base', home)
    clicks, response = [], None
    task = skills.place_near_ally(ctx, 'tesla_reactor', (945, 100), max_probes=2)
    while True:
        try: step = task.send(response)
        except StopIteration: break
        response = None
        if isinstance(step, Look): response = step.fn(world.snapshot(), None)
        elif isinstance(step, Poll): response = step.reason in ('placement', 'camera at the target')
        elif isinstance(step, Wait): response = step.until(world.snapshot(), None)
        elif isinstance(step, Act):
            clicks.extend((ctx.camera, s) for s in step.steps if s.kind == 'click')
        else: raise AssertionError(step)
    construction_clicks = [(camera, s.point) for camera, s in clicks
                           if s.button == 'left' and s.point[0] < LAYOUT.tactical.w]
    assert len(construction_clicks) == 1
    assert construction_clicks[0][0] == ('away' if valid else 'home')
    assert ctx.camera == 'home' and world.structures['tesla_reactor'] == 4


@pytest.mark.parametrize('faction', ['allied', 'soviet'])
def test_strategy_has_no_fixed_country_and_optional_units_require_availability(faction):
    book = playbook.load(playbook.PROFILE.with_name(f'startlux-ra2-{faction}.json'))
    assert book.data['faction'] == faction and 'country' not in book.data
    assert 'placement' in book.questions
    if faction == 'allied':
        from addons.chronodivide.tests.test_allied import game
        world, army, reflex = game()
        assert 'grand_cannon' not in reflex.unit_options('defenses', ['pillbox', 'patriot'], 6000, 500)[0]
    else:
        from addons.chronodivide.tests.test_high_tech import advantaged
        world, army, reflex, time = advantaged()
        assert 'desolator' not in reflex.unit_options('infantry', ['conscript'], 6000, time[0])[0]
