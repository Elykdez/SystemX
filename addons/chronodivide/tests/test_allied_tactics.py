from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide import allied_tactics as tactics, playbook
from addons.chronodivide.perception.unit_labels import read_label
from addons.chronodivide.perception.unitboxes import Unit
from addons.chronodivide.tests.test_strategy import setup, run, LAYOUT, attack_setup, CameraGame, online_radar

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'profiles' / 'startlux-ra2-allied.json'
FIXTURES = Path(__file__).with_name('fixtures')


def view(**changes):
    state = dict(center=(400, 400), target=(600, 400), threat=None,
                 close=False, hurt=False, in_range=True, covered=True)
    state.update(changes)
    return state


def test_native_mirage_tooltip_does_not_become_a_prism_guess():
    frame = cv2.imread(str(FIXTURES / 'allied-mirage-tooltip-native.png'))
    assert read_label(frame, (788, 269)) == 'mirage_tank'
    assert read_label(frame, (300, 600)) is None
    assert read_label(None, (788, 269)) is None


def test_native_prism_tooltip_is_verified_without_lowering_the_label_threshold():
    frame = cv2.imread(str(FIXTURES / 'allied-prism-tooltip-native.png'))
    assert read_label(frame, (397, 436)) == 'prism_tank'
    assert read_label(frame, (100, 100)) is None


def test_unknown_no_tooltip_views_never_identify_a_prism():
    for name in ('allied-prism-ready-without-count.png', 'allied-factory-healthy.png'):
        frame = cv2.imread(str(FIXTURES / name))
        for point in ((100, 100), (425, 368), (700, 650)):
            assert read_label(frame, point) is None


@pytest.mark.parametrize('state,clear,expected', [
    (None, True, ['hold']),
    (view(close=True, threat=(430, 400)), True, ['reposition', 'hold']),
    (view(hurt=True), True, ['reposition', 'hold']),
    (view(threat=(700, 400)), False, ['hold', 'bombard']),
    (view(in_range=False), False, ['hold']),
    (view(in_range=False), True, ['hold', 'advance']),
    (view(in_range=False, threat=(700, 400)), True, ['hold']),
    (view(target=None, in_range=False), True, ['hold']),
    (view(covered=False), True, ['hold']),
    (view(target=None, in_range=False, route=(900, 400)), False, ['hold']),
    (view(target=None, in_range=False, route=(900, 400), covered=False), True, ['hold']),
])
def test_advance_requires_repeated_clear_views_and_bombard_requires_safe_range(state, clear, expected):
    assert tactics.choices(state, clear) == expected


def test_isometric_standoff_does_not_move_into_the_enemy_building():
    plan = playbook.load(BOOK)['battle']['prism']
    target = (300, 300)
    for home in ((800, 300), (600, 650)):
        point = tactics.offset(target, home, plan['range_pixels'], LAYOUT.tactical)
        assert abs(tactics.separation(target, point) - plan['range_pixels']) < 2
        assert tactics.separation(point, home) < tactics.separation(target, home)
    assert tactics.offset((35, 35), (0, 0), plan['range_pixels'], LAYOUT.tactical) is None


def test_escort_types_take_separate_flanks_outside_the_prism_advance_corridor():
    plan = playbook.load(BOOK)['battle']['prism']
    points = [tactics.escort_screen((400, 400), (1000, 400), i, plan, LAYOUT.tactical) for i in range(2)]
    assert points == [(475, 325), (475, 475)]
    assert all(abs(2 * (point[1] - 400)) >= 150 for point in points)
    assert tactics.escort_screen((400, 40), (1000, 40), 0, plan, LAYOUT.tactical) == (475, 40)


def test_staging_uses_the_enemy_perimeter_not_a_building_behind_the_front(monkeypatch):
    from addons.chronodivide import army
    blobs = [(200, (250, 300), (0, 0, 20, 20)), (200, (400, 300), (0, 0, 20, 20))]
    monkeypatch.setattr(tactics, 'enemy_blobs', lambda *a, **k: blobs)
    monkeypatch.setattr(army, 'building_point', lambda canvas, blob, **k: blob[1])
    plan = playbook.load(BOOK)['battle']['prism']
    point = tactics.staging_point(np.zeros((768, 1024, 3), np.uint8), (1200, 300), LAYOUT.tactical, plan)
    assert point == (730, 300)
    assert all(tactics.separation(point, b[1]) >= plan['range_pixels'] for b in blobs)
    blobs.append((50, point, (0, 0, 10, 10)))
    assert tactics.staging_point(np.zeros((768, 1024, 3), np.uint8), (1200, 300), LAYOUT.tactical, plan) is None


def test_native_building_fragments_do_not_trigger_a_prism_retreat():
    from addons.chronodivide.client import ClientProfile
    from addons.chronodivide.perception import colors
    previous = ClientProfile.load('chronodivide').use_colors()
    try:
        frame = cv2.imread(str(FIXTURES / 'enemy-base-siege.png'))
        prism = Unit((680, 110, 720, 150), 'ours', 'vehicle', 1, (682, 100, 36))
        plan = playbook.load(BOOK)['battle']['prism']
        scene = tactics.picture(frame, LAYOUT.tactical, [prism], plan)
        assert scene['target'] is not None and scene['in_range']
        assert scene['threat'] is None and not scene['close']
        hurt = Unit(prism.box, 'ours', 'vehicle', 0.65, prism.bar)
        assert tactics.picture(frame, LAYOUT.tactical, [hurt], plan)['hurt']
    finally:
        colors.use(*previous)


def test_native_tesla_fragments_allow_only_a_repeated_safe_standoff_step():
    from addons.chronodivide.client import ClientProfile
    from addons.chronodivide.perception import colors
    from addons.chronodivide.perception.unitboxes import our_units
    previous = ClientProfile.load('chronodivide-allied').use_colors()
    try:
        frame = cv2.imread(str(FIXTURES / 'prism-tesla-standoff-native.png'))
        prisms = [u for u in our_units(frame, compressed=True) if u.kind == 'vehicle']
        plan = playbook.load(BOOK)['battle']['prism']
        scene = tactics.picture(frame, LAYOUT.tactical, prisms, plan)
        scene['covered'] = True
        point = scene['guarded_advance']
        assert len(prisms) == 6 and scene['threat'] is not None
        assert not scene['close'] and not scene['hurt'] and not scene['in_range']
        assert point is not None
        assert tactics.separation(point, scene['target']) < plan['range_pixels']
        assert tactics.separation(point, scene['threat']) > plan['close_pixels']
        assert tactics.choices(scene, False) == ['hold']
        assert tactics.choices(scene, True) == ['hold', 'advance']
    finally:
        colors.use(*previous)


@pytest.mark.parametrize('quality', [None, 95, 90, 80])
def test_damaged_tesla_base_supplies_a_target_without_discarding_threat_fragments(quality):
    from addons.chronodivide.client import ClientProfile
    from addons.chronodivide.perception import colors
    previous = ClientProfile.load('chronodivide-allied').use_colors()
    try:
        frame = cv2.imread(str(FIXTURES / 'damaged-tesla-hold-native.png'))
        if quality is not None:
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        prisms = [u for u in tactics.our_units(frame, compressed=True) if u.kind == 'vehicle']
        scene = tactics.picture(frame, LAYOUT.tactical, prisms, playbook.load(BOOK)['battle']['prism'])
        assert len(prisms) == 7
        assert scene['target'] == (235, 288) and scene['in_range']
        assert scene['threat'] is not None and not scene['close']
        scene['covered'] = True
        assert tactics.choices(scene, False) == ['hold', 'bombard']
        # A mirrored palette makes the same sprite friendly, never an enemy target.
        colors.use(colors.ENEMIES[0], [colors.OWN])
        assert tactics.tesla_coil_targets(frame, (0, 0, 850, 736), compressed=True) == []
    finally:
        colors.use(*previous)


def test_damaged_tesla_match_abstains_on_other_native_scenes():
    from addons.chronodivide.client import ClientProfile
    from addons.chronodivide.perception import colors
    previous = ClientProfile.load('chronodivide-allied').use_colors()
    try:
        for path in FIXTURES.glob('*.png'):
            if path.name == 'damaged-tesla-hold-native.png':
                continue
            frame = cv2.imread(str(path))
            assert tactics.tesla_coil_targets(frame, (0, 0, min(850, frame.shape[1]), min(736, frame.shape[0]))) == []
    finally:
        colors.use(*previous)


def test_standoff_advance_is_rejected_if_it_would_enter_an_attackers_close_radius(monkeypatch):
    from addons.chronodivide import army
    blobs = [(200, (800, 400), (780, 380, 40, 40)), (90, (700, 400), (690, 390, 20, 20))]
    monkeypatch.setattr(tactics, 'enemy_blobs', lambda *a, **k: blobs)
    monkeypatch.setattr(army, 'building_point', lambda canvas, blob, **k: blob[1])
    prism = Unit((380, 370, 420, 410), 'ours', 'vehicle', 1, (382, 375, 36))
    scene = tactics.picture(np.zeros((768, 1024, 3), np.uint8), LAYOUT.tactical, [prism],
                            playbook.load(BOOK)['battle']['prism'])
    scene['covered'] = True
    assert scene['threat'] == (700, 400) and not scene['close']
    assert scene['guarded_advance'] is None
    assert tactics.choices(scene, True) == ['hold']


def test_small_native_tesla_foundation_is_a_firing_target_without_discarding_its_threat_pixels():
    from addons.chronodivide.client import ClientProfile
    from addons.chronodivide.perception import colors
    from addons.chronodivide.perception.unitboxes import our_units
    previous = ClientProfile.load('chronodivide-allied').use_colors()
    try:
        frame = cv2.imread(str(FIXTURES / 'prism-small-tesla-foundation-native.png'))
        prisms = [u for u in our_units(frame, compressed=True) if u.kind == 'vehicle']
        scene = tactics.picture(frame, LAYOUT.tactical, prisms, playbook.load(BOOK)['battle']['prism'])
        scene['covered'] = True
        assert len(prisms) == 7
        assert scene['target'][0] < 140 and scene['target'][1] > 220
        assert scene['threat'] is not None and not scene['close'] and scene['in_range']
        assert tactics.choices(scene, False) == ['hold', 'bombard']
    finally:
        colors.use(*previous)


def test_a_close_flat_vehicle_cluster_still_blocks_prism_fire(monkeypatch):
    from addons.chronodivide import army
    blob = (90, (450, 400), (430, 393, 45, 15))
    monkeypatch.setattr(tactics, 'enemy_blobs', lambda *a, **k: [blob])
    monkeypatch.setattr(army, 'building_point', lambda canvas, blob, **k: blob[1])
    prism = Unit((380, 370, 420, 410), 'ours', 'vehicle', 1, (382, 375, 36))
    scene = tactics.picture(np.zeros((768, 1024, 3), np.uint8), LAYOUT.tactical, [prism],
                            playbook.load(BOOK)['battle']['prism'])
    scene['covered'] = True
    assert scene['target'] is not None and scene['threat'] is not None and scene['close']
    assert tactics.choices(scene, True) == ['reposition', 'hold']


def test_staging_camera_leaves_room_on_the_home_side_of_the_enemy():
    radar = online_radar()
    home, target = radar.own[0][1], radar.enemy[0][1]
    plan = playbook.load(BOOK)['battle']['prism']
    camera = tactics.approach_camera(radar, target, home, LAYOUT.tactical, plan)
    assert camera != target
    assert tactics.separation(camera, home) < tactics.separation(target, home)
    pixels = tactics.separation(radar.to_tactical(camera, LAYOUT.tactical), radar.to_tactical(target, LAYOUT.tactical))
    assert abs(pixels - plan['range_pixels']) < 60  # Integer radar coordinates quantize the shift.


def test_siege_camera_follows_the_verified_prisms_instead_of_an_empty_enemy_view():
    from addons.chronodivide.army import battle_session
    world, ctx, runner, army, reflex, executor, time = setup()
    army.book = playbook.load(BOOK)
    world.units_built['prism_tank'] = 4
    radar = online_radar()
    world.set('radar', radar)
    ctx.radar_home = radar.own[0][1]
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.target = radar.enemy[0][1]
    campaign.prism_roles = {'prism_tank': '5', 'mirage_tank': '6'}
    task = battle_session(ctx, campaign, army=army)
    first = next(task)
    keys = [step.keys for step in first.steps if step.kind == 'key']
    assert keys == [('5',), ('5',)]
    assert not any(step.kind == 'click' and step.button == 'left' for step in first.steps)
    task.close()


def test_verified_prism_focus_keeps_a_bounded_window_after_confirming_the_new_view(monkeypatch):
    from addons.chronodivide.army import battle_session
    from addons.chronodivide.tests.test_unitboxes import draw_bar
    from addons.chronodivide.tasks import Act
    world, ctx, runner, army, reflex, executor, time = setup()
    army.book = playbook.load(BOOK)
    world.units_built['prism_tank'] = 4
    world.set('radar', online_radar())
    ctx.radar_home = world.radar.value.own[0][1]
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    draw_bar(ctx.last_canvas, 400, 340, 36)
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.target = world.radar.value.enemy[0][1]
    campaign.prism_roles = {'prism_tank': '5', 'mirage_tank': '6'}
    rounds = []
    def order(*a):
        rounds.append(True)
        yield Act([])
        return []
    monkeypatch.setattr(tactics, 'round_order', order)
    runner.start('army:focus', 'army', battle_session(ctx, campaign, army=army, seconds=30, every=3))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=300)
    assert len(rounds) == 3


def execute(monkeypatch, states, broker=None, roles=None, route=False, move=None, clear=True):
    world, ctx, runner, army, reflex, executor, time = setup()
    book = playbook.load(BOOK)
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    unit = Unit((380, 380, 420, 420), 'ours', 'vehicle', 1, (382, 375, 36))
    monkeypatch.setattr(tactics, 'selected', lambda ctx: [unit])
    frames = iter(states)
    last = states[-1]
    monkeypatch.setattr(tactics, 'picture', lambda *a: next(frames, last))
    army.campaign.prism_roles = roles if roles is not None else {'prism_tank': '5', 'mirage_tank': '6', 'ifv': '7'}
    army.campaign.prism_role_wave = army.campaign.waves
    army.campaign.prism_probe_at = time[0]
    army.campaign.prism_clear_since = time[0] - 5 if clear else time[0]
    if route:
        radar = online_radar()
        world.set('radar', radar)
        army.campaign.active = True
        army.campaign.kind = 'enemy'
        army.campaign.target = radar.enemy[0][1]
        army.campaign.prism_view = radar.view
    if move is not None:
        army.campaign.prism_move = (*move, (army.campaign.target, army.campaign.waves))
    runner.start('army:prism', 'army', tactics.round_order(ctx, army.campaign, book, broker))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    return [s for act in executor.acts for s in act[2]], world.events


@pytest.mark.parametrize('name,expected', [
    ('allied-clipped-enemy-base-native.jpg', {'Left'}),
    ('allied-revealed-enemy-base-native.jpg', {None, 'Up'}),
])
def test_native_edge_peek_reveals_clipped_enemies_without_pushing_formation_out_of_view(name, expected):
    from addons.chronodivide.client import ClientProfile
    from addons.chronodivide.perception import colors
    from addons.chronodivide.perception.unitboxes import our_units

    previous = ClientProfile.load('chronodivide-allied').use_colors()
    try:
        canvas = cv2.imread(str(FIXTURES / name))
        units = our_units(canvas, compressed=True)
        assert units
        assert tactics.reveal_edge(canvas, LAYOUT.tactical, units) in expected
    finally:
        colors.use(*previous)


def test_camera_peek_rechecks_the_scene_before_ordering_prisms(monkeypatch):
    monkeypatch.setattr(tactics, 'reveal_edge', lambda *a: 'Left')
    steps, events = execute(monkeypatch, [view(target=None, in_range=False), view()])
    pan = next(i for i, step in enumerate(steps) if step[0] == 'key' and step[2] == ('Left',))
    attack = next(i for i, step in enumerate(steps) if step[0] == 'click' and step[2] == ('ctrl',))
    assert pan < attack and events[-1]['choice'] == 'bombard'
    assert sum(step[0] == 'key' and step[2] == ('Left',) for step in steps) == 1


def test_edge_peek_ignores_wide_game_messages(monkeypatch):
    unit = Unit((380, 380, 420, 420), 'ours', 'vehicle', 1, (380, 375, 36))
    monkeypatch.setattr(tactics, 'enemy_blobs', lambda *a, **k: [(200, (90, 20), (0, 10, 180, 10))])
    assert tactics.reveal_edge(np.zeros((768, 1024, 3), np.uint8), LAYOUT.tactical, [unit]) is None


def test_only_prisms_receive_the_direct_building_attack(monkeypatch):
    steps, events = execute(monkeypatch, [view()])
    attacks = [s for s in steps if s[0] == 'click' and s[2] == ('ctrl',)]
    assert len(attacks) == 1 and attacks[0][1] == (600, 400)
    attack_index = steps.index(attacks[0])
    assert [s[2] for s in steps[:attack_index] if s[0] == 'key'][-1] == ('5',)
    assert not any(s[2] == ('ctrl', 'shift') for s in steps)
    assert events[-1]['choice'] == 'bombard'


def test_clear_travel_view_advances_toward_the_known_enemy_without_firing(monkeypatch):
    steps, events = execute(monkeypatch, [view(target=None, in_range=False)], route=True)
    assert events[-1]['choice'] == 'advance'
    assert events[-1]['facts']['target'] == 'route'
    assert any(s[0] == 'click' and s[3] == 'left' and s[2] == ('alt',) for s in steps)
    assert not any(s[0] == 'click' and 'ctrl' in s[2] for s in steps)


def test_repeated_stationary_travel_uses_bounded_longer_pathfinding_steps():
    from addons.chronodivide.army import Campaign
    campaign = Campaign()
    campaign.target = (900, 100)
    campaign.prism_roles = {'prism_tank': '5'}
    points = []
    for _ in range(5):
        point, campaign.prism_travel = tactics.travel_step(
            campaign, (400, 400), (1000, 400), LAYOUT.tactical, 90, (1, 2, 3, 4), 3)
        points.append(point)
    assert points == [(490, 400), (490, 400), (580, 400), (670, 400), (670, 400)]


@pytest.mark.parametrize('change', ['progress', 'camera', 'count', 'target', 'wave', 'roles'])
def test_travel_recovery_resets_after_progress_or_observation_identity_changes(change):
    from addons.chronodivide.army import Campaign
    campaign = Campaign()
    center, marker, count = (400, 400), (1, 2, 3, 4), 3
    for _ in range(4):
        _, campaign.prism_travel = tactics.travel_step(
            campaign, center, (1000, 400), LAYOUT.tactical, 90, marker, count)
    if change == 'progress':
        center = (420, 400)
    elif change == 'camera':
        marker = (2, 2, 4, 4)
    elif change == 'count':
        count = 2
    elif change == 'target':
        campaign.target = (901, 100)
    elif change == 'wave':
        campaign.waves += 1
    else:
        campaign.prism_roles = {'prism_tank': '8'}
    point, _ = tactics.travel_step(campaign, center, (1000, 400), LAYOUT.tactical, 90, marker, count)
    assert tactics.separation(center, point) == 90


def test_longer_travel_falls_back_to_an_in_bounds_endpoint():
    from addons.chronodivide.army import Campaign
    campaign = Campaign()
    for _ in range(5):
        point, campaign.prism_travel = tactics.travel_step(
            campaign, (600, 400), (1000, 400), LAYOUT.tactical, 90, None, 3)
    assert point == (780, 400)


def test_an_accepted_move_continues_while_the_group_is_moving_then_releases_when_stationary():
    from addons.chronodivide.army import Campaign
    campaign = Campaign()
    marker = (1, 2, 3, 4)
    campaign.prism_move = (marker, (400, 400), False, (campaign.target, campaign.waves))
    assert tactics.travelling(campaign, (440, 400), marker)
    assert tactics.travelling(campaign, (480, 400), marker)
    assert tactics.travelling(campaign, (400, 400), (2, 2, 4, 4))
    assert not tactics.travelling(campaign, (400, 400), (2, 2, 4, 4))
    assert campaign.prism_move is None


@pytest.mark.parametrize('clear', [False, True])
def test_safe_pending_travel_is_not_replaced_or_stopped_by_a_new_camera_check(monkeypatch, clear):
    steps, events = execute(monkeypatch, [view(center=(440, 400), target=None, in_range=False)],
                            route=True, move=(online_radar().view, (400, 400), False), clear=clear)
    assert events[-1]['continuing']
    assert not any(name.startswith('prism_') for name, point in events[-1]['orders'])
    assert not any(s[0] == 'key' and s[2] == ('s',) for s in steps)


def test_an_explicit_model_hold_stops_a_pending_move(monkeypatch):
    class Model:
        clock = staticmethod(lambda: 1000)
        generation = staticmethod(lambda: 0)
        def submit(self, decision):
            decision.on_result(decision, {'prism_battle': {'choice': 'hold'}}, 'systemone', {})
    steps, events = execute(monkeypatch, [view(center=(440, 400), target=None, in_range=False)],
                            Model(), route=True, move=(online_radar().view, (400, 400), False))
    assert not events[-1]['continuing']
    assert events[-1]['choice'] == 'hold'
    assert any(s[0] == 'key' and s[2] == ('s',) for s in steps)


def test_a_distant_building_approach_finishes_its_accepted_move_before_retargeting(monkeypatch):
    steps, events = execute(monkeypatch, [view(center=(440, 400), target=(800, 400), in_range=False)],
                            move=(None, (400, 400), False))
    assert events[-1]['continuing']
    assert not any(name.startswith('prism_') for name, point in events[-1]['orders'])


def test_new_danger_interrupts_a_pending_route_move(monkeypatch):
    steps, events = execute(monkeypatch, [view(close=True, threat=(440, 400), target=None, in_range=False)],
                            route=True, move=(online_radar().view, (400, 400), False))
    assert not events[-1]['continuing']
    assert events[-1]['choice'] == 'reposition'


def test_close_attackers_trigger_prism_retreat_and_only_escort_defense(monkeypatch):
    steps, events = execute(monkeypatch, [view(close=True, threat=(440, 400))])
    assert not any(s[0] == 'click' and s[2] == ('ctrl',) for s in steps)
    assert any(s[0] == 'click' and s[1] == (310, 400) and s[2] == ('alt',) for s in steps)
    defenses = [s for s in steps if s[2] == ('ctrl', 'shift')]
    assert len(defenses) == 2 and all(s[1] == (440, 400) for s in defenses)
    assert events[-1]['choice'] == 'reposition'


@pytest.mark.parametrize('route', [False, True])
def test_new_threat_rejects_a_delayed_model_advance(monkeypatch, route):
    class Model:
        clock = staticmethod(lambda: 1000)
        generation = staticmethod(lambda: 0)
        def submit(self, decision):
            assert decision.kind == 'tactical'
            assert 'advance' in decision.questions['prism_battle']['criteria']
            decision.on_result(decision, {'prism_battle': {'choice': 'advance'}}, 'systemone', {})
    steps, events = execute(monkeypatch, [view(target=None if route else (600, 400), in_range=False),
                                        view(close=True, threat=(440, 400))], Model(), route=route)
    assert events[-1]['choice'] == 'hold'
    assert not any(s[0] == 'click' and s[1] == (490, 400) for s in steps)


def test_missing_prisms_hold_visible_units_without_stopping_unseen_arrivals(monkeypatch):
    steps, events = execute(monkeypatch, [None], roles={'mirage_tank': '6'})
    assert not any(s[0] == 'click' and s[3] == 'left' for s in steps)
    assert any(s[0] == 'key' and s[2] == ('s',) for s in steps)
    assert not any(s[0] == 'key' and s[2] == ('1',) for s in steps)
    assert sum(s[0] == 'key' and s[2] == ('p',) for s in steps) == 1
    assert events[-1]['choice'] == 'hold'


@pytest.mark.parametrize('alive,edge,teams', [(True, False, 1), (False, False, 1),
                                            (True, True, 1), (True, False, 2)])
def test_offscreen_escorts_use_centered_proof_and_fresh_prism_coordinates(monkeypatch, alive, edge, teams):
    from addons.chronodivide.tests.test_strategy import Recorder
    world, ctx, runner, army, reflex, executor, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    campaign = army.campaign
    campaign.active = True
    campaign.kind = 'enemy'
    world.set('radar', online_radar())
    campaign.target = world.radar.value.enemy[0][1]
    campaign.prism_roles = {'prism_tank': '5', 'mirage_tank': '6'}
    if teams == 2:
        campaign.prism_roles['ifv'] = '7'
    campaign.prism_role_wave = campaign.waves
    campaign.prism_clear_since = time[0] - 5
    campaign.prism_view = world.radar.value.view
    state = dict(camera='5', selection='5', visits=0, remote_reads=0)
    def prism():
        x = 10 if edge and state['visits'] >= 2 else 300 + 30 * state['visits']
        y = 400 + 10 * state['visits']
        return Unit((x-20, y-30, x+20, y+10), 'ours', 'vehicle', 1, (x-18, y-25, 36))
    remote = Unit((680, 70, 720, 110), 'ours', 'vehicle', 1, (682, 75, 36))
    class NativeGroups(Recorder):
        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            keys = [s.keys[0] for s in steps if s.kind == 'key' and s.keys[0] in ('5', '6', '7')]
            if keys:
                state['selection'] = keys[-1]
                if len(keys) == 2:
                    state['camera'] = keys[-1]
                    state['visits'] += int(keys[-1] == '5')
    runner.executor = executor = NativeGroups()
    def selected(ctx):
        if state['selection'] != state['camera']:
            return []
        if state['selection'] == '5':
            return [prism()]
        state['remote_reads'] += 1
        return [remote] if alive else []
    monkeypatch.setattr(tactics, 'selected', selected)
    monkeypatch.setattr(tactics, 'picture', lambda *a: view(center=tactics.ground(prism()), target=None, in_range=False))
    seen = []
    def edge_view(canvas, tactical, units):
        seen.extend(tactics.ground(u) for u in units)
        return None
    monkeypatch.setattr(tactics, 'reveal_edge', edge_view)
    runner.start('army:focus', 'army', tactics.round_order(ctx, campaign, playbook.load(BOOK), None))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=400)
    steps = [s for act in executor.acts for s in act[2]]
    moves = [s[1] for s in steps if s[0] == 'click' and s[2] == ('alt',)]
    assert moves == ([(360, 420), (420, 440)][:teams] if alive and not edge else [])
    assert ('mirage_tank' in campaign.prism_roles) is alive
    assert state['remote_reads'] >= (1 if alive else 2)
    decision = [e for e in world.events if e.get('kind') == 'prism_decision'][-1]
    assert decision['choice'] == 'hold' and decision['facts']['cover'] == 'missing'
    assert seen == [tactics.ground(prism())]
    assert not any(s[0] == 'chord' or 'ctrl' in s[2] for s in steps)


def test_missing_escort_triggers_local_verification_with_existing_prisms(monkeypatch):
    from addons.chronodivide.tasks import Act
    world, ctx, runner, army, reflex, executor, time = setup()
    campaign = army.campaign
    campaign.prism_roles = {'prism_tank': '5'}
    campaign.prism_role_wave = campaign.waves
    called = []
    def form(ctx, campaign, plan):
        called.append(True)
        yield Act([])
    monkeypatch.setattr(tactics, 'form_roles', form)
    task = tactics.round_order(ctx, campaign, playbook.load(BOOK), None)
    next(task)
    assert called
    task.close()


@pytest.mark.parametrize('alive', [False, True])
def test_empty_general_group_checks_exclusive_siege_groups_before_declaring_a_loss(monkeypatch, alive):
    from addons.chronodivide import army as army_module
    from addons.chronodivide.tests.test_strategy import Recorder
    from addons.chronodivide.tests.test_unitboxes import draw_bar
    world, ctx, runner, army, reflex, executor, time = setup()
    army.book = playbook.load(BOOK)
    world.units_built['prism_tank'] = 4
    world.set('radar', online_radar())
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.target = world.radar.value.enemy[0][1]
    campaign.prism_roles = {'prism_tank': '5', 'mirage_tank': '6'}
    army.failures = 1
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    class ExclusiveGroups(Recorder):
        def act(self, name, lane, steps, commands=1):
            super().act(name, lane, steps, commands)
            for step in steps:
                if step.kind == 'key' and step.keys[0] in ('1', '5', '6'):
                    ctx.last_canvas[:] = 0
                    if step.keys == ('5',) and alive:
                        for x in (350, 400, 450):
                            draw_bar(ctx.last_canvas, x, 340, 36)
    runner.executor = ExclusiveGroups()
    def jump(ctx, point, *a):
        if False:
            yield
        return True
    monkeypatch.setattr(army_module, 'radar_jump', jump)
    runner.start('army:verify', 'army', army._verify_army())
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    assert campaign.loss_confirmed is (not alive)
    assert not campaign.depleted_confirmed
    verified = [e for e in world.events if e.get('kind') == 'army_verified'][-1]
    assert verified['units'] == ([3, 3] if alive else [0, 0])


def test_soviet_keeps_its_existing_battle_policy():
    assert playbook.default()['battle'].get('prism') is None


def test_allied_attack_focus_waits_for_prism_production_even_with_a_large_basic_army():
    from addons.chronodivide.tests.test_allied import game
    world, army, reflex = game()
    world.set('radar', online_radar())
    world.units_built['grizzly_tank'] = 40
    assert 'attack' not in reflex.feasible_postures(500)
    world.units_built['prism_tank'] = 1
    assert 'attack' in reflex.feasible_postures(500)


@pytest.mark.parametrize('miss', [False, True])
def test_type_group_is_rebuilt_from_visible_verified_units_only(monkeypatch, miss):
    world, ctx, runner, army, reflex, executor, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    units = [Unit((x, 380, x + 40, 420), 'ours', 'vehicle', 1, (x, 375, 36)) for x in (380, 480)]
    selection = iter([units, units[:1], units, [] if miss else units[:1], units, units])
    monkeypatch.setattr(tactics, 'selected', lambda ctx: next(selection, []))
    monkeypatch.setattr(tactics, 'read_label', lambda *a: 'prism_tank')
    runner.start('roles', 'army', tactics.form_roles(ctx, army.campaign, playbook.load(BOOK)['battle']['prism']))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    steps = [s for act in executor.acts for s in act[2]]
    assert ('prism_tank' in army.campaign.prism_roles) == (not miss)
    # Tooltip delay belongs to the hover transaction, so another lane cannot park the pointer first.
    assert any([s[0] for s in act[2]] == ['move', 'move', 'wait'] for act in executor.acts)
    assert any(s[0] == 'chord' and s[2] == ('ctrl', '5') for s in steps) == (not miss)
    if not miss:
        typed = next(i for i, s in enumerate(steps) if s[0] == 'key' and s[2] == ('t',))
        saved = next(i for i, s in enumerate(steps) if s[0] == 'chord' and s[2] == ('ctrl', '5'))
        between = steps[typed + 1:saved]
        assert any(s[0] == 'click' and s[3] == 'right' for s in between)
        assert any(s[0] == 'click' and s[1] == tactics.ground(units[1]) and s[2] == ('shift',) for s in between)


def test_field_role_probes_preserve_the_front_camera_instead_of_centering_the_map_army():
    world, ctx, runner, army, reflex, executor, time = setup()
    ctx.camera = 'away'
    task = tactics.form_roles(ctx, army.campaign, playbook.load(BOOK)['battle']['prism'])
    first = next(task)
    assert [s.keys for s in first.steps if s.kind == 'key'] == [('p',), ('s',)]
    assert not any(s.kind == 'chord' for s in first.steps)
    task.close()


def test_role_probes_center_and_stop_the_formation_before_reading_units(monkeypatch):
    world, ctx, runner, army, reflex, executor, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    units = [Unit((380, 380, 420, 420), 'ours', 'vehicle', 1, (380, 375, 36))]
    selection = iter([units, units, units, units, units])
    group_keys = []
    stopped = []
    original = executor.act

    def act(name, lane, steps, commands=1):
        original(name, lane, steps, commands)
        group_keys.extend(s for s in steps if s.kind == 'key' and s.keys == ('1',))
        stopped.extend(s for s in steps if s.kind == 'key' and s.keys == ('s',))

    monkeypatch.setattr(executor, 'act', act)
    monkeypatch.setattr(tactics, 'selected', lambda ctx: next(selection, [])
                        if len(group_keys) >= 2 and stopped else [])
    monkeypatch.setattr(tactics, 'read_label', lambda *a: 'ifv')
    runner.start('roles', 'army', tactics.form_roles(ctx, army.campaign, playbook.load(BOOK)['battle']['prism']))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    assert army.campaign.prism_roles == {'ifv': '7'}
    assert len(group_keys) == 3  # center twice, then reselect after stopping only visible military
    assert len(stopped) == 1


@pytest.mark.parametrize('edge', ['left', 'right', 'top', 'bottom'])
def test_role_verification_excludes_edge_scroll_units_from_probes_and_type_groups(monkeypatch, edge):
    world, ctx, runner, army, reflex, executor, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    tactical = ctx.layout.tactical
    point = {'left': (tactical.x + 5, 400), 'right': (tactical.x + tactical.w - 5, 400),
             'top': (400, tactical.y + 5), 'bottom': (400, tactical.y + tactical.h - 5)}[edge]
    clipped = Unit((point[0] - 20, point[1] - 20, point[0] + 20, point[1] + 20),
                   'ours', 'vehicle', 1, (point[0] - 18, point[1] - 25, 36))
    inside = Unit((380, 380, 420, 420), 'ours', 'vehicle', 1, (380, 375, 36))
    selection = iter([[clipped, inside], [inside], [clipped, inside], [inside], [inside]])
    monkeypatch.setattr(tactics, 'selected', lambda ctx: next(selection, []))
    inspected = []
    monkeypatch.setattr(tactics, 'read_label', lambda canvas, at: inspected.append(at) or 'prism_tank')
    runner.start('roles', 'army', tactics.form_roles(ctx, army.campaign, playbook.load(BOOK)['battle']['prism']))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    assert army.campaign.prism_roles == {'prism_tank': '5'}
    assert inspected and set(inspected) == {tactics.ground(inside)}
    assert not any(s[0] in ('click', 'move') and s[1] == point
                   for act in executor.acts for s in act[2])


@pytest.mark.parametrize('failure', ['empty', 'wrong_type'])
def test_an_unverified_saved_group_is_not_reported_as_a_prism_role(monkeypatch, failure):
    world, ctx, runner, army, reflex, executor, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    units = [Unit((380, 380, 420, 420), 'ours', 'vehicle', 1, (380, 375, 36))]
    selection = iter([units, units, units, units, [] if failure == 'empty' else units])
    labels = iter(['prism_tank', 'prism_tank', 'prism_tank', 'ifv'])
    monkeypatch.setattr(tactics, 'selected', lambda ctx: next(selection, []))
    monkeypatch.setattr(tactics, 'read_label', lambda *a: next(labels, None))
    runner.start('roles', 'army', tactics.form_roles(ctx, army.campaign, playbook.load(BOOK)['battle']['prism']))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    assert army.campaign.prism_roles == {}
    assert not any(e['kind'] == 'prism_role' for e in world.events)


def test_bounded_role_retries_reach_new_units_beyond_the_first_probe_batch(monkeypatch):
    world, ctx, runner, army, reflex, executor, time = setup()
    plan = playbook.load(BOOK)['battle']['prism']
    units = [Unit((x, 380, x + 40, 420), 'ours', 'vehicle', 1, (x, 375, 36))
             for x in range(20, 20 + 20 * (plan['label_probes'] + 1), 20)]
    inspected = []
    monkeypatch.setattr(tactics, 'selected', lambda ctx: units)
    monkeypatch.setattr(tactics, 'read_label', lambda canvas, point: inspected.append(point))
    for attempt in range(2):
        runner.start('roles', 'army', tactics.form_roles(ctx, army.campaign, plan))
        assert run(runner, world, time, lambda: not runner.busy('army'), ticks=300)
        assert len(inspected) == (attempt + 1) * plan['label_probes']
    assert tactics.ground(units[-1]) in inspected
    assert army.campaign.prism_roles == {}  # inspecting a candidate does not verify its identity


def test_correct_selection_count_does_not_hide_a_wrong_vehicle_type(monkeypatch):
    world, ctx, runner, army, reflex, executor, time = setup()
    ctx.last_canvas = np.zeros((768, 1024, 3), np.uint8)
    units = [Unit((x, 380, x + 40, 420), 'ours', 'vehicle', 1, (x, 375, 36)) for x in (380, 480)]
    selection = iter([units, units[:1], units, units[:1], units])
    labels = iter(['prism_tank', 'prism_tank', 'prism_tank', 'mirage_tank'])
    monkeypatch.setattr(tactics, 'selected', lambda ctx: next(selection, []))
    monkeypatch.setattr(tactics, 'read_label', lambda *a: next(labels, None))
    runner.start('roles', 'army', tactics.form_roles(ctx, army.campaign, playbook.load(BOOK)['battle']['prism']))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=200)
    assert army.campaign.prism_roles == {}
    assert not any(s[0] == 'chord' and s[2] == ('ctrl', '5') for act in executor.acts for s in act[2])


@pytest.mark.parametrize('verified', [False, True])
def test_first_wave_forms_roles_at_home_before_camera_or_movement(monkeypatch, verified):
    from addons.chronodivide.army import attack_wave
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=16)
    runner.executor = executor = CameraGame(world)
    cameras = []
    def form(ctx, campaign, plan):
        cameras.append(ctx.camera)
        if verified:
            campaign.prism_roles.update(prism_tank='5', mirage_tank='6')
        if False:
            yield
    monkeypatch.setattr(tactics, 'form_roles', form)
    monkeypatch.setattr(tactics, 'staging_point', lambda *a: (700, 350))
    army.campaign.active = True  # The scheduler marks this before ticking the first coroutine.
    target = world.radar.value.enemy[0][1]
    runner.start('attack', 'army', attack_wave(ctx, target, 'enemy', campaign=army.campaign,
                 prism=playbook.load(BOOK)['battle']['prism']))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=100)
    assert cameras == ['home']
    assert army.campaign.waves == int(verified)
    steps = [s for act in executor.acts for s in act[2]]
    assert any(s[0] == 'click' and s[1] == (700, 350) and s[2] == ('ctrl', 'shift') for s in steps) == verified
    assert not any(s[0] == 'click' and s[2] == ('ctrl',) for s in steps)


def test_prism_staging_stops_if_the_camera_box_disappears_after_a_jump(monkeypatch):
    from dataclasses import replace
    from addons.chronodivide import army as army_module
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=16)
    army.campaign.prism_roles = {'prism_tank': '5', 'mirage_tank': '6'}
    def jump(ctx, point):
        world.set('radar', replace(world.radar.value, view=None))
        if False:
            yield
        return True
    monkeypatch.setattr(army_module, 'radar_jump', jump)
    target = world.radar.value.enemy[0][1]
    runner.start('attack', 'army', army_module.attack_wave(ctx, target, 'enemy', campaign=army.campaign,
                 prism=playbook.load(BOOK)['battle']['prism']))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=100)
    assert runner.finished[-1].status == 'done'
    assert army.campaign.waves == 0
    assert not any(s[0] == 'click' and s[2] for act in executor.acts for s in act[2])
    assert any(e.get('reason') == 'standoff view unavailable' for e in world.events)


@pytest.mark.parametrize('safe', [False, True])
def test_prism_reinforcements_preserve_exclusive_groups_until_staging_is_verified(monkeypatch, safe):
    from addons.chronodivide import army as army_module
    world, ctx, runner, army, reflex, executor, time = attack_setup(units=16)
    campaign = army.campaign
    campaign.active = campaign.team_formed = True
    campaign.waves = 2
    campaign.prism_roles = {'prism_tank': '5', 'mirage_tank': '6'}
    calls = []
    def jump(ctx, point):
        calls.append('camera')
        if False:
            yield
        return True
    def field(ctx):
        calls.append('merge')
        if False:
            yield
        return True
    points = iter([(700, 350), (600, 350)] if safe else [None])
    monkeypatch.setattr(army_module, 'radar_jump', jump)
    monkeypatch.setattr(army_module, 'field_team', field)
    monkeypatch.setattr(tactics, 'staging_point', lambda *a: next(points))
    target = world.radar.value.enemy[0][1]
    runner.start('attack', 'army', army_module.attack_wave(ctx, target, 'enemy', campaign=campaign,
                 keep_home_infantry=True, prism=playbook.load(BOOK)['battle']['prism']))
    assert run(runner, world, time, lambda: not runner.busy('army'), ticks=100)
    assert calls == (['camera', 'merge', 'camera'] if safe else ['camera'])
    assert campaign.prism_staging_failed is (not safe)
    assert campaign.waves == (3 if safe else 2)
    if safe:
        assert not campaign.prism_roles
        assert any(s[0] == 'click' and s[1] == (600, 350) and s[2] == ('ctrl', 'shift')
                   for act in executor.acts for s in act[2])
    else:
        assert campaign.prism_roles == {'prism_tank': '5', 'mirage_tank': '6'}
