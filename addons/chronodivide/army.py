"""Army control on its own task lane: guard the base, defend it, or attack in waves.

Orders use verified bindings: P selects the military units on screen and a second P extends
the selection across the map (the game says "Selected across SCREEN", then "... MAP");
Ctrl+Shift+left click is the attack-move under the classic left-click scheme (units fight
whatever they meet on the way), and a right click on the map clears the selection.

The posture comes from standing intent (System One's `posture` answer, or the rule default), and so
does the attack size: how many tanks must be at home before an attack starts (System One's
`attack_size` answer). The numbers of waves and battle sessions are the System One profile's
(playbook sections army, attack, battle, defense); the ones left here are about reading the
screen and driving the input.

gather  The army stays home. Enemies seen near the base get a defense session: P twice selects
        owned military across the map and attack-moves toward the nearest raiders.
defend  Like gather, and enemies seen near home on the radar are hunted down even when
        the camera does not show them.
attack  The radar says where to go. With nothing selected, a radar click moves the camera
        there (with a selection it would order a plain move instead). The whole army is
        selected with P and attack-moved onto the largest enemy object in view (or the spot
        the radar pointed at), and H brings the camera home. Waves repeat when enough new
        units are ready and at a fixed interval, so reinforcements join and survivors are
        re-targeted as enemy buildings fall. Where no enemy has been seen yet, the likely
        enemy start positions (our base mirrored through the map center) are tried first,
        then the most shrouded areas.

Team control and battle sessions (vision): a reinforcing wave saves the whole army as team 1
(P twice, Ctrl+1); other waves command team 1 only. Between waves the camera stays with the
fight for a battle session (the profile's battle section: 12 s, an order every 1.6 s): it jumps to
our largest group at the front by a radar click, and each round selects team 1 (1), finds our
units by their health bars (perception/unitboxes.py) and attack-moves toward nearby enemy units,
or context-clicks the nearest building when only buildings are near. House-color centroids
locate an area, not a verified object hitbox: Ctrl+click on them can force fire on empty ground.
A session ends early
when a finished building waits for placement or the base is attacked; production goes on
meanwhile. Without the radar, 1 pressed twice centers the camera on the team instead.

Scouting and looks (System One's intel answer, see intel.py): a scout wave sends the few early
units with a plain move to the likely enemy start, and brings them home once the radar
shows the enemy base; a look moves only the camera to the enemy base or to our fight,
surveys the view, and returns.

Sightings near the base are reported to the world as 'attack' events, which makes the
defend posture available to System One.
"""

from collections import Counter
import math
import time

from . import miners, patrol, playbook
from .executor import Step
from .intel import survey
from .perception.radar import distance
from .perception.terrain import ground_mask, open_spots
from .perception.enemy_threats import deployed_gi_targets, standing_gi_targets
from .perception.unitboxes import our_units, recognize
from .perception.units import enemy_blobs, house_blobs
from .perception import colors
from .tasks import Act, Look, Poll

ATTACK_MOVE = ('ctrl', 'shift')
# A contextual click can select a vehicle overlapping the building instead of attacking.
FORCE_ATTACK = ('ctrl',)
# Select the whole army wherever it is: P (on screen), then P again (across the map).
# The game applies a selection on its next frame: wait before the order click.
SELECT_ALL = [Step.key('p'), Step.wait(80), Step.key('p'), Step.wait(150)]
HOME_RADIUS = 14                                            # radar pixels around home that count as "our base"
BASE_DOTS = 12                                              # an enemy dot group this large holds buildings
CLOSE_DOTS = 10                                             # smaller groups are fought only this close to the army
FIELD_RADIUS = 20                                           # our dots this close to the target: the army is there
LOST_AFTER = 20.0                                           # seconds without them there: the attack failed
TEAM = '1'                                                  # the field army's team (Ctrl+1 at each wave)
FOCUS_MISSES = 2                                            # focuses that did not move the camera: give up teams
BUILDING_PIXELS = 100                                       # enemy blobs this large are buildings (tanks: 30-80)
FIGHT_PIXELS = 60                                           # enemy unit pixels near the army worth fighting
UNIT_SHARE = 0.3                                            # ... and their share of the enemy pixels seen


def combat_units(world, book=None):
    """Army strength produced so far in tanks: unit cost over the profile's strength unit (a
    Rhino), infantry weighed as the profile says (at half value six Conscripts count as a third
    of a tank). Counts confirmed or probably queued units; losses are not seen."""
    plan = (book or playbook.default())['army']
    items = world.knowledge['items']
    value = sum(n * items[item]['cost'] * (plan['infantry_weight'] if items[item]['lane'] == 'infantry' else 1.0)
                for item, n in world.units_built.items()
                if items.get(item, {}).get('lane') in ('infantry', 'vehicles') and item not in plan['not_combat'])
    return int(value / items[plan['strength_unit']]['cost'] + 1e-9)


def combat_count(world, book=None):
    """How many combat units were produced (any kind, one each)."""
    plan = (book or playbook.default())['army']
    items = world.knowledge['items']
    return sum(n for item, n in world.units_built.items()
               if items.get(item, {}).get('lane') in ('infantry', 'vehicles') and item not in plan['not_combat'])


def threat(canvas, min_blob=25, min_total=50, reach=160, compressed=False):
    """(point, pixels) of the largest enemy group in view, or None.

    Blobs of the opponent's house color closer than `reach` to the largest one form the
    group; the target is its area-weighted center. Tiny blobs (effects, stray pixels) are ignored.
    """
    if canvas is None:
        return None
    blobs = [b for b in enemy_blobs(canvas, compressed=compressed) if b[0] >= min_blob]
    if sum(b[0] for b in blobs) < min_total:
        return None
    seed = blobs[0][1]
    group = [b for b in blobs if abs(b[1][0] - seed[0]) + abs(b[1][1] - seed[1]) <= reach]
    area = sum(b[0] for b in group)
    x = sum(b[0] * b[1][0] for b in group) / area
    y = sum(b[0] * b[1][1] for b in group) / area
    return (round(x), round(y)), area


def choose_target(radar, home, current=None, cleared=(), known=None):
    """(radar point, kind) for the next attack wave, or (None, None).

    Enemy buildings come first (large groups of dots): the group that is large and close to our
    army in the field (or to home, when the army is at home). Small groups (units) are fought
    only when they are right beside the army or our base; farther ones are left to the
    attack-move on the way (live game 12: chasing roaming units kept the army mid-map for three
    minutes, never reaching the unexplored enemy base). Then the remembered enemy base
    (`known`) while dots remain there. Once it is empty, whatever the radar still shows: a
    short game is won only when every enemy building is destroyed. Otherwise the current guess
    stays until the radar shows it explored; then the next unexplored mirror guess, then the
    most shrouded area.
    """
    if radar is None or not radar.online:
        return None, None
    anchor = radar.army_center(home) or home or radar.center
    buildings = [e for e in radar.enemy if e[0] >= BASE_DOTS]
    near = [e for e in radar.enemy if distance(e[1], anchor) <= CLOSE_DOTS or
            (home is not None and distance(e[1], home) <= HOME_RADIUS)]
    candidates = buildings or near
    if candidates:
        # Finish the current hostile base before switching to another opponent's larger base.
        continuing = [e for e in buildings if current is not None and distance(e[1], current) < 8]
        if continuing:
            return max(continuing)[1], 'enemy'
        area, point = max(candidates, key=lambda e: math.sqrt(e[0]) / (1 + distance(e[1], anchor) / 15))
        return point, 'enemy'
    if known is not None:
        if radar.enemy_near(known, CLOSE_DOTS):
            return tuple(known), 'enemy'
        if radar.enemy:                  # the base is razed: whatever is left, largest and nearest first
            area, point = max(radar.enemy, key=lambda e: math.sqrt(e[0]) / (1 + distance(e[1], anchor) / 15))
            return point, 'enemy'
    if (current is not None and radar.explored_near(current) < 0.6 and
            not any(area >= BASE_DOTS and distance(current, point) < HOME_RADIUS
                    for area, point in radar.allies) and
            not any(distance(current, c) < 8 for c in cleared)):
        return current, 'guess'
    for guess in radar.guesses(home):
        if radar.explored_near(guess) < 0.6 and not any(distance(guess, c) < 8 for c in cleared):
            return guess, 'guess'
    spot = radar.unexplored_spot(home)
    return (spot, 'search') if spot else (None, None)


def target_in_view(canvas, radar, radar_point, tactical, reach=260, compressed=False):
    """Tactical point to attack-move to once the camera shows the target area."""
    aim = radar.to_tactical(radar_point, tactical) if radar is not None and radar.in_view(radar_point) \
        else (tactical.x + tactical.w // 2, tactical.y + tactical.h // 2)
    blobs = [b for b in enemy_blobs(canvas, compressed=compressed) if b[0] >= 25] if canvas is not None else []
    near = [b for b in blobs if abs(b[1][0] - aim[0]) + abs(b[1][1] - aim[1]) <= reach]
    if near:
        aim = near[0][1]                  # the largest enemy object near the spot (buildings are large)
    margin = 30
    return (min(max(aim[0], tactical.x + margin), tactical.x + tactical.w - margin),
            min(max(aim[1], tactical.y + margin), tactical.y + tactical.h - margin))


class _Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def contains(self, point):
        return self.x <= point[0] < self.x + self.w and self.y <= point[1] < self.y + self.h


MESSAGES = _Box(0, 0, 160, 64)          # game messages ("Selected across MAP") are red text up here


def _standing_raider(canvas, tactical, center, reach, margin=20):
    if canvas is None:
        return None
    region = (tactical.x, tactical.y, tactical.w, tactical.h)
    points = [p for p in standing_gi_targets(canvas, region, compressed=True)
              if not MESSAGES.contains(p) and abs(p[0] - center[0]) + abs(p[1] - center[1]) <= reach and
              tactical.x + margin <= p[0] <= tactical.x + tactical.w - margin and
              tactical.y + margin <= p[1] <= tactical.y + tactical.h - margin]
    return min(points, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]), default=None)


def building_point(canvas, blob, compressed=False):
    """Pick a low visible surface nearer the building's ground footprint than its roof."""
    _, (cx, _), (x, y, w, h) = blob
    ys, xs = colors.enemy_mask(canvas[y:y + h, x:x + w], compressed=compressed).nonzero()
    if not len(xs):
        return None
    # Chrono Divide resolves building clicks through the ground tile below the sprite.
    index = ((xs + x - cx) ** 2 + (ys - h + 1) ** 2).argmin()
    return int(xs[index] + x), int(ys[index] + y)


def siege_target(canvas, tactical, aim, reach=300, building=100, ours=2, compressed=False):
    """The enemy building to attack when our units are already beside it, or None.

    Attack-move makes units fight the enemies they meet, but arrived units stand idle next to
    plain buildings (live game 7: Rhinos parked inside the enemy base). A direct click on a
    building orders the whole selection to attack it. Buildings are the large blobs of the
    enemy color; our units there show as blobs of our color.
    """
    if canvas is None:
        return None
    if compressed and _standing_raider(canvas, tactical, aim, reach) is not None:
        return None
    region = (tactical.x, tactical.y, tactical.w, tactical.h)
    if any(abs(p[0] - aim[0]) + abs(p[1] - aim[1]) <= reach
           for p in deployed_gi_targets(canvas, region, compressed=compressed)):
        return None                         # fight confirmed defenders before overriding with siege
    margin = 30                         # a building at the view's edge is left for a later view
    targets = [b for b in enemy_blobs(canvas, region, compressed=compressed) if b[0] >= building and
               abs(b[1][0] - aim[0]) + abs(b[1][1] - aim[1]) <= reach and
               tactical.x + margin <= b[1][0] <= tactical.x + tactical.w - margin and
               tactical.y + margin <= b[1][1] <= tactical.y + tactical.h - margin]
    if not targets:
        return None
    # The building nearest the spot our units were sent to: they need not walk around the base.
    target = min(targets, key=lambda b: abs(b[1][0] - aim[0]) + abs(b[1][1] - aim[1]))
    point = building_point(canvas, target, compressed)
    if point is None:
        return None
    own = [b for b in house_blobs(canvas, region) if b[0] >= 12 and not MESSAGES.contains(b[1]) and
           abs(b[1][0] - point[0]) + abs(b[1][1] - point[1]) <= 2 * reach]
    return point if len(own) >= ours else None


def battle_order(canvas, tactical, center, reach=330, building_reach=700, compressed=False):
    """The next order for the field army from a view of the fight, or None.

    ('fight', point): enemy units are near the army: attack-move toward the nearest enemy area.
    The game acquires a unit target; a color centroid alone cannot identify its hitbox.
    ('siege', point): only enemy buildings are near: the one nearest to the army (arrived units
    stand idle beside plain buildings). Game messages at the top left are not enemies.
    """
    if canvas is None:
        return None
    region = (tactical.x, tactical.y, tactical.w, tactical.h)
    defenders = [p for p in deployed_gi_targets(canvas, region, compressed=compressed)
                 if abs(p[0] - center[0]) + abs(p[1] - center[1]) <= reach]
    if defenders:
        return 'fight', min(defenders, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))
    if compressed:
        standing = _standing_raider(canvas, tactical, center, reach)
        if standing is not None:
            return 'fight', standing
    margin = 30
    seen = [b for b in enemy_blobs(canvas, region, compressed=compressed) if b[0] >= 25 and not MESSAGES.contains(b[1]) and
            tactical.x + margin <= b[1][0] <= tactical.x + tactical.w - margin and
            tactical.y + margin <= b[1][1] <= tactical.y + tactical.h - margin]
    # Units are fought near the army; buildings anywhere in the view (resumed game 28: the base
    # stood 400 px from the army's center and every session found nothing to do).
    near = [b for b in seen if abs(b[1][0] - center[0]) + abs(b[1][1] - center[1]) <= reach]
    units = [b for b in near if b[0] < BUILDING_PIXELS]
    unit_pixels = sum(b[0] for b in units)
    # Buildings show small house-color parts too (the enemy base fixture: 1000 small pixels beside
    # 4600 in buildings, no units), so enemy units must be a real share of what is seen.
    if unit_pixels >= FIGHT_PIXELS and unit_pixels >= UNIT_SHARE * sum(b[0] for b in near):
        nearest = min(units, key=lambda b: abs(b[1][0] - center[0]) + abs(b[1][1] - center[1]))
        return 'fight', nearest[1]
    buildings = [b for b in seen if b[0] >= BUILDING_PIXELS and
                 abs(b[1][0] - center[0]) + abs(b[1][1] - center[1]) <= building_reach]
    if buildings:
        nearest = min(buildings, key=lambda b: abs(b[1][0] - center[0]) + abs(b[1][1] - center[1]))
        point = building_point(canvas, nearest, compressed)
        return ('siege', point) if point is not None else None
    return None


def camera_moved(radar, before, tolerance=2):
    """The camera box on the radar left the position it had before a focus key."""
    if radar is None or not radar.online or radar.view is None or before is None:
        return False
    now = radar.view_center
    return abs(now[0] - before[0]) + abs(now[1] - before[1]) > tolerance


def _compressed_raider_groups(canvas, tactical, center, reach, margin=20):
    """Coherent JPEG infantry fragments shared by threat confirmation and targeting."""
    if canvas is None:
        return []
    region = (tactical.x, tactical.y, tactical.w, tactical.h)
    fragments = [b for b in enemy_blobs(canvas, region, min_area=3, close=3, compressed=True)
                 if not MESSAGES.contains(b[1]) and
                 abs(b[1][0] - center[0]) + abs(b[1][1] - center[1]) <= reach and
                 tactical.x + margin <= b[1][0] <= tactical.x + tactical.w - margin and
                 tactical.y + margin <= b[1][1] <= tactical.y + tactical.h - margin]
    groups = []
    for seed in fragments:
        group = [b for b in fragments if abs(b[1][0] - seed[1][0]) + abs(b[1][1] - seed[1][1]) <= 80]
        area = sum(b[0] for b in group)
        if len(group) >= 3 and area >= 12:
            point = (round(sum(b[0] * b[1][0] for b in group) / area),
                     round(sum(b[0] * b[1][1] for b in group) / area))
            groups.append((area, point))
    return groups


def raider_target(canvas, tactical, center, reach, margin=20, compressed=False):
    """The enemy nearest to `center` in the view (within `reach`), as a point on it, or None.

    At home every enemy in view is a raider: its size does not matter (raiders side by side
    showed as one 194 px blob, above the building size, and the first live defense session fired
    no shot). The enemy color is split with little merging, so the point lies on one unit."""
    if canvas is None:
        return None
    region = (tactical.x, tactical.y, tactical.w, tactical.h)
    deployed = [p for p in deployed_gi_targets(canvas, region, compressed=compressed)
                if not MESSAGES.contains(p) and abs(p[0] - center[0]) + abs(p[1] - center[1]) <= reach and
                tactical.x + margin <= p[0] <= tactical.x + tactical.w - margin and
                tactical.y + margin <= p[1] <= tactical.y + tactical.h - margin]
    if deployed:
        return min(deployed, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))
    if compressed:
        standing = _standing_raider(canvas, tactical, center, reach, margin)
        if standing is not None:
            return standing
    near = [b for b in enemy_blobs(canvas, region, min_area=8, close=3, compressed=compressed)
            if not MESSAGES.contains(b[1]) and abs(b[1][0] - center[0]) + abs(b[1][1] - center[1]) <= reach and
            tactical.x + margin <= b[1][0] <= tactical.x + tactical.w - margin and
            tactical.y + margin <= b[1][1] <= tactical.y + tactical.h - margin]
    if not near and compressed:
        # Standing infantry can retain only 3-5 blue pixels each in JPEG. A coherent
        # group is enough for attack-move; isolated specks cannot establish a target.
        near = _compressed_raider_groups(canvas, tactical, center, reach, margin)
    if not near:
        return None
    return min(near, key=lambda b: abs(b[1][0] - center[0]) + abs(b[1][1] - center[1]))[1]


def defense_session(ctx, rounds, every, reach, home=True, keep_home_infantry=False, team_formed=False):
    """Fight raiders at home: select defenders and attack-move toward the nearest enemy area.

    The game acquires actual targets even if a moving sprite leaves its measured color point.
    """
    world, tactical = ctx.world, ctx.layout.tactical
    compressed = getattr(ctx, 'compressed_frames', False)
    middle = (tactical.x + tactical.w // 2, tactical.y + tactical.h // 2)
    orders, defenders = [], 0
    if not home and keep_home_infantry and not team_formed:
        radar = world.radar.value
        if radar is None or radar.view is None:
            return
        x0, y0, x1, y1 = radar.view
        point = ((x0 + x1) // 2, (y0 + y1) // 2)
        if not (yield from field_team(ctx, allow_remote=True)) or not (yield from radar_jump(ctx, point)):
            return
        ctx.camera = 'away'
    yield Act([Step.click(ctx.clear_point, 'right')] + ([Step.key('h')] if home else [])
              + [Step.move(ctx.park)], commands=0)
    if home:
        ctx.camera = 'home'
    yield Poll(lambda s, v: False, 0.2)
    for index in range(rounds):
        if index:
            yield Poll(lambda s, v: False, every)
        target = yield Look(lambda s, v: raider_target(ctx.last_canvas, tactical, middle, reach, compressed=compressed))
        if target is None:
            break                                            # no raider in view any more
        selection = SELECT_ALL
        if not home and keep_home_infantry:
            selection = [Step.key(TEAM), Step.wait(150)]
        yield Act([Step.click(ctx.clear_point, 'right')] + selection + [Step.move(ctx.park)],
                  commands=0)
        yield Poll(lambda s, v: False, 0.1)
        ours = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                          if ctx.last_canvas is not None else [])
        defenders = len(ours)
        # Keep checking the threatened view while map-selected defenders travel here.
        target = yield Look(lambda s, v: raider_target(ctx.last_canvas, tactical, middle, reach,
                                                     compressed=compressed))
        if target is None:
            break
        # Other lanes can clear selection while we inspect the fresh frame. Select and
        # order in one transaction so distant defenders still receive the recall.
        yield Act([Step.click(ctx.clear_point, 'right')] + selection +
                  [Step.click(target, modifiers=ATTACK_MOVE), Step.move(ctx.park)], commands=1)
        orders.append(target)
    yield Act([Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=0)   # nothing stays selected
    world.note('order', order='defense', orders=len(orders), defenders=defenders,
               selection='field_team' if not home and keep_home_infantry else 'map_military',
               at=[list(p) for p in orders[-3:]])


def radar_jump(ctx, radar_point, timeout=0.6):
    """Camera to a radar point: a right click first (with a selection, the radar click would be a
    move order), the click, the cursor parked on the radar (see Context.radar_park), and a wait
    until the camera box shows the point. True when it does."""
    world = ctx.world
    yield Act([Step.click(ctx.clear_point, 'right'), Step.click(radar_point), Step.move(ctx.radar_park)],
              commands=0)
    return (yield Poll(lambda s, v: world.radar.value is not None and world.radar.value.in_view(radar_point),
                       timeout, 'camera at the target'))


def group_scouts(ctx, army, team, count, tries=3):
    """Save the dog scouts as their own team: H, P (the units at home, only the dogs while other
    infantry waits for this), and Ctrl+team when the health bars show infantry only: all of the
    dogs, or as many as are out after a few tries (a dog still training misses the team)."""
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h')], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    yield Act([Step.key('p'), Step.wait(150), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.12)
    units = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                       if ctx.last_canvas is not None else [])
    enough = len(units or []) == count or army.scout_tries >= tries
    if units and all(u.kind == 'infantry' for u in units) and len(units) <= count and enough:
        yield Act([Step.chord('ctrl', team), Step.wait(60)], commands=0)
        army.scouts_ready = True
    yield Act([Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=0)
    ctx.world.note('scouts', grouped=army.scouts_ready, seen=len(units or []))


def selected_team(ctx, team):
    """A fresh home-view observation of one saved team, without moving its units."""
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    yield Act([Step.key(team), Step.wait(150), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.12)
    units = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                       if ctx.last_canvas is not None else [])
    yield Act([Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=0)
    return units or []


def explore_route(ctx, army, team, route, leg):
    """Queue a finite route once; background observations must verify departure and return."""
    button = ctx.layout.points.get('waypoint')
    if button is None or army.exploring or army.explore_failed:
        return
    units = yield from selected_team(ctx, team)
    if not units or any(u.kind != 'infantry' for u in units):
        return
    tactical, margin = ctx.layout.tactical, 40
    hx, hy = ctx.home
    plan = army.book['intel']['explore']
    dx, dy = plan['return_offset']
    home = (min(max(hx + dx, tactical.x + margin), tactical.x + tactical.w - margin),
            min(max(hy + dy, tactical.y + margin), tactical.y + tactical.h - margin))
    points = [(min(max(hx + dx, tactical.x + margin), tactical.x + tactical.w - margin),
               min(max(hy + dy, tactical.y + margin), tactical.y + tactical.h - margin)) for dx, dy in route]
    points = list(dict.fromkeys(p for p in points if p != home))[:15] + [home]
    if army._recent_threat():
        return
    yield Act([Step.key(team), Step.wait(120), Step.key('s'), Step.route(points, button),
               Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=len(points))
    army.exploring, army.explore_departed = True, False
    army.explore_home, army.explore_count = home, len(units)
    army.explore_origin = tuple(u.center for u in units)
    # Recorded dogs needed about 110 seconds for seven nodes, including pathfinding detours.
    deadline = plan['deadline']
    army.explore_deadline = army.clock() + min(deadline['max_seconds'], max(
        deadline['min_seconds'], len(points) * leg * deadline['leg_multiplier']))
    army.explore_checked = army.clock()
    ctx.world.note('order', order='explore', waypoints=len(points), finite=True, arrived=False)


def monitor_exploration(ctx, army):
    units = yield from selected_team(ctx, army.book['intel']['scout_team'])
    army.explore_checked = army.clock()
    if not army.exploring or not units:
        return
    near = all(distance(u.center, army.explore_home) <= 100 for u in units)
    if not near and any(min(distance(u.center, p) for p in army.explore_origin) > 50 for u in units):
        army.explore_departed = True
    elif (near and army.explore_departed and len(units) == army.explore_count and
          all(u.kind == 'infantry' and u.health is not None and
              u.health >= army.book['intel']['explore']['return_min_health'] for u in units)):
        army.exploring, army.explored = False, True
        ctx.world.note('explore', arrived=True, verified=len(units))


def monitor_patrol(ctx, army):
    state = army.patrol
    plan = army.book['patrol']
    units = yield from selected_team(ctx, plan['team'])
    now = army.clock()
    army.patrol_checked = now
    if not state.active:
        return
    expected = Counter(u.kind for u in state.members)
    healthy = [u.health for u in units if u.health is not None]
    same = Counter(u.kind for u in units) == expected and len(healthy) == len(state.members)
    health = min(healthy) if healthy else None
    previous = min((u.health for u in state.members if u.health is not None), default=1)
    # A partial camera view is not fresh damage; repeated missing members must not renew defense.
    damaged = same and health is not None and health < previous - plan['damage_drop']
    army._look(now)                       # points come from this frame, never the last route
    contact = army.seen is not None or army.home_threat is not None
    if contact or damaged:
        army.threat_at = now              # defense rechecks the screen before issuing an order
        ctx.world.note('patrol', contact=contact, damaged=bool(damaged), seen=len(units))
    action = patrol.contact_action(state, contact, health, army.book)
    if action == 'retreat':
        yield Act([Step.key(plan['team']), Step.wait(120), Step.click(ctx.front_point(reach=plan['retreat_reach'])),
                   Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=1)
        state.active = False
        return
    if not same:
        army.patrol_verified = False
        return                           # absence from a view alone does not establish a death
    state.members = tuple(units)
    army.patrol_verified, army.patrol_health = True, health
    if (not contact and not damaged and health >= plan['min_health'] and now - state.last_order >= plan['refresh_every'] and
            now - army.census_at <= army.book['army']['census']['fresh_within'] and
            army.census_healthy_vehicles >= plan['reserve_vehicles'] + plan['vehicles']):
        point = army._scout_point(now)
        approaches = [ctx.world.radar.value.to_tactical(point, ctx.layout.tactical)] if point is not None else []
        yield from patrol.order_patrol(ctx, state, patrol.guard_route(ctx, approaches, army.book),
                                      at_home=True, book=army.book)


def field_team(ctx, allow_remote=False):
    """Keep home infantry on guard; defense can recall military outside the home view."""
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h')], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    yield Act(SELECT_ALL + [Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.12)
    if ctx.last_canvas is None:
        return False
    infantry = yield Look(lambda s, v: [u for u in our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                       if u.kind == 'infantry'] if ctx.last_canvas is not None else [])
    home_infantry = len(infantry)
    # Units can move while input is paced. Re-read the selection after each click.
    for attempt in range(home_infantry * 2):
        if not infantry:
            break
        unit = infantry[attempt % len(infantry)]
        # Native infantry is selected at its ground tile, 25 pixels below the health bar.
        # The sprite center can hit empty ground and order the entire selection to move.
        point = (unit.bar[0] + unit.bar[2] // 2, unit.bar[1] + 25)
        yield Act([Step.click(point, modifiers=('shift',)), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.12)
        infantry = yield Look(lambda s, v: [u for u in our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                           if u.kind == 'infantry'])
    if infantry:
        ctx.world.note('field_team', formed=False, reason='home infantry still selected')
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        return False
    vehicles = yield Look(lambda s, v: [u for u in our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                       if u.kind == 'vehicle'])
    if not vehicles and not allow_remote:
        ctx.world.note('field_team', formed=False, reason='no vehicles selected')
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        return False
    yield Act([Step.chord('ctrl', TEAM), Step.wait(60)], commands=0)
    ctx.world.note('field_team', formed=True, home_infantry=home_infantry,
                   visible_vehicles=len(vehicles), remote_selection=allow_remote and not vehicles)
    return True


def attack_wave(ctx, radar_point, kind, order='attack', on_view=None, campaign=None, reinforce=True, select=None,
                keep_home_infantry=False, prism=None):
    """Camera to the target by a radar click, the whole army attack-moves there, camera home.

    When our units are already beside an enemy building there, the order is a direct attack
    on that building instead (a siege; `campaign.sieging` then asks for the next one sooner).
    order 'move' sends a plain move instead (scouts should arrive, not fight on the way);
    order 'look' gives no order at all. `on_view(canvas, radar)` sees the target view.

    In a campaign, a reinforcing wave selects the whole army (P twice) and saves it as team 1;
    other waves select team 1 only, so new units wait at home until enough can go together
    (one at a time they were destroyed on the way).
    """
    world, tactical = ctx.world, ctx.layout.tactical
    defer_team = (prism is not None and campaign is not None and campaign.team_formed
                  and keep_home_infantry and order == 'attack' and select is None and reinforce)
    if (keep_home_infantry and order == 'attack' and select is None and
            not defer_team and (campaign is None or reinforce or not campaign.team_formed or not campaign.teams)):
        if not (yield from field_team(ctx)):
            return
        if campaign is not None:
            campaign.teams = True
            if prism is not None:
                campaign.prism_roles.clear()  # Native control groups have exclusive membership.
        select = [Step.key(TEAM), Step.wait(150)]
    if select is not None:
        pass                                                 # the caller's selection (the scout team)
    elif campaign is None or not campaign.teams:
        select = SELECT_ALL
    elif reinforce or not campaign.team_formed:
        select = SELECT_ALL + [Step.chord('ctrl', TEAM), Step.wait(60)]
    else:
        select = [Step.key(TEAM), Step.wait(150)]

    if prism is not None and order == 'attack' and campaign is not None and not campaign.waves:
        from .allied_tactics import form_roles
        if 'prism_tank' not in campaign.prism_roles or len(campaign.prism_roles) < 2:
            yield from form_roles(ctx, campaign, prism)
        campaign.prism_role_wave = campaign.waves + 1
        campaign.prism_probe_at = world.clock()
        if 'prism_tank' not in campaign.prism_roles or len(campaign.prism_roles) < 2:
            yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
            world.note('prism_staging', sent=False, reason='Prism and escort type groups are not both verified')
            return

    def radar_now():
        return world.radar.value

    ctx.camera = 'away'
    point, home, siege = radar_point, False, None
    staged = False
    try:
        view_point = radar_point
        if prism is not None and order == 'attack':
            from .allied_tactics import approach_camera
            view_point = approach_camera(radar_now(), radar_point, ctx.radar_home, tactical, prism)
        moved = yield from radar_jump(ctx, view_point)
        if moved:
            yield Poll(lambda s, v: False, 0.12)            # let the new view render
            if on_view is not None:
                yield Look(lambda s, v: on_view(ctx.last_canvas, radar_now()))
            if order != 'look':
                point = yield Look(lambda s, v: target_in_view(ctx.last_canvas, radar_now(), radar_point, tactical,
                                                              compressed=getattr(ctx, 'compressed_frames', False)))
            if order == 'attack':
                siege = yield Look(lambda s, v: siege_target(ctx.last_canvas, tactical, point,
                                                            compressed=getattr(ctx, 'compressed_frames', False)))
        if prism is not None and order == 'attack' and moved and ctx.radar_home is not None:
            current_radar = radar_now()
            if current_radar is None or not current_radar.online or current_radar.view is None:
                ctx.world.note('prism_staging', sent=False, reason='standoff view unavailable')
                return
            from .allied_tactics import staging_point
            home_point = current_radar.to_tactical(ctx.radar_home, tactical)
            approach = yield Look(lambda s, v: staging_point(ctx.last_canvas, home_point, tactical, prism))
            if approach is not None and defer_team:
                # A failed staging lookup must not erase the existing siege groups.
                if not (yield from field_team(ctx)):
                    return
                campaign.prism_roles.clear()
                ctx.camera = 'away'
                if not (yield from radar_jump(ctx, view_point)):
                    return
                yield Poll(lambda s, v: False, 0.12)
                current_radar = radar_now()
                if current_radar is None or not current_radar.online or current_radar.view is None:
                    return
                home_point = current_radar.to_tactical(ctx.radar_home, tactical)
                approach = yield Look(lambda s, v: staging_point(ctx.last_canvas, home_point, tactical, prism))
                select = [Step.key(TEAM), Step.wait(150)]
            if approach is not None:
                point, siege = approach, None
                yield Act(select + [Step.click(point, modifiers=ATTACK_MOVE)], commands=2)
                ctx.world.note('prism_staging', sent=True, target=list(point))
                staged = True
            else:
                ctx.world.note('prism_staging', sent=False, reason='no visible standoff point')
                return
        elif prism is not None and order == 'attack':
            ctx.world.note('prism_staging', sent=False, reason='standoff view unavailable')
            return
        elif siege is not None:                                # a direct attack on the building
            point = siege
            yield Act(select + [Step.click(point, modifiers=FORCE_ATTACK)], commands=2)
        elif order != 'look':                                # without the camera: order on the radar
            yield Act(select + [Step.click(point, modifiers=ATTACK_MOVE if order == 'attack' else ())],
                      commands=2)
        if campaign is not None:
            campaign.sieging = siege is not None
            campaign.team_formed = campaign.team_formed or (campaign.teams and order != 'look')
            if order != 'look':
                sent_at = world.clock()
                campaign.waves += 1
                campaign.last_wave = sent_at
                if campaign.kind == 'guess' and campaign.guess_since is None:
                    campaign.guess_since = sent_at
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
        home = True
        yield Poll(lambda s, v: False, 0.2)
    finally:
        if prism is not None and order == 'attack' and campaign is not None:
            campaign.prism_staging_failed = campaign.team_formed and not staged
        # Interrupted (a pause) before H: the camera is somewhere else until H is pressed again.
        ctx.camera = 'home' if home else 'lost'
    world.note('order', order=f'{order}_wave' if order != 'look' else 'look', target=list(radar_point), aim=kind,
               camera=bool(moved), at=list(point), siege=siege is not None)


def front_point(radar, home, target=None):
    """Radar point of our largest group away from home, preferring groups near the target: where
    the fight is. None when everything is at home or the radar is off."""
    if radar is None or not radar.online:
        return None
    away = [(area, point) for area, point in radar.own if home is None or distance(point, home) > HOME_RADIUS]
    if target is not None:
        away = [(a, p) for a, p in away if distance(p, target) <= FIELD_RADIUS] or away
    return max(away)[1] if away else None


def army_center(units):
    """The middle of our units on screen (the median of their boxes' centers), or None."""
    if not units:
        return None
    xs = sorted(u.center[0] for u in units)
    ys = sorted(u.center[1] for u in units)
    return xs[len(xs) // 2], ys[len(ys) // 2]


def battle_session(ctx, campaign, intel=None, leave=None, seconds=None, every=None, stay=False, army=None):
    """Watch the fight and command it for `seconds`, an order every `every` s (by default the
    profile's battle section; see the module doc), then H.

    `leave(snapshot)` says when to go home early (a building to place, the base attacked). The
    camera goes to the front by a radar click with nothing selected (with a selection that click
    would be a move order). Without field dots, 1 pressed twice selects the team and centers the
    camera on it; that centers on the whole team, which is mid-map while reinforcements travel
    (live game 24: 41 of 51 such views showed empty ground), so it is only the fallback. Without
    radar, a double tap with neither camera movement nor visible selected units counts as a miss;
    after FOCUS_MISSES misses the campaign stops using teams. Online teams stay tracked for loss
    verification. A team already in view needs no camera movement."""
    world, tactical = ctx.world, ctx.layout.tactical
    plan = playbook.default()['battle']
    seconds = plan['seconds'] if seconds is None else seconds
    every = plan['order_every'] if every is None else every
    radar = world.radar.value
    cached_front = (campaign.confirmed_front if campaign.front_seen_at is not None and
                    0 <= world.clock() - campaign.front_seen_at <= LOST_AFTER else None)
    front = front_point(radar, ctx.radar_home, campaign.target) or cached_front
    prism_team = (campaign.prism_roles.get('prism_tank') if army is not None and
                  army.prism_plan() is not None and campaign.active and campaign.team_formed else None)
    if prism_team:
        # Follow the siege units in transit; an empty destination hides intervening fights.
        front = None
    elif front is None and campaign.active and radar is not None and radar.online:
        front = campaign.target              # inspect the target; a team centroid can be empty ground
    if (front is None and radar is not None and radar.online and
            not (campaign.active and campaign.teams and campaign.team_formed)):
        return                           # no established campaign team to inspect
    before = radar.view_center if radar is not None and radar.online else None
    ctx.camera = 'away'
    home, moved, orders, seen_units = False, False, [], []
    try:
        if front is not None:
            moved = yield from radar_jump(ctx, front, 0.7)
        else:
            focus_team = prism_team or TEAM
            yield Act([Step.click(ctx.clear_point, 'right'), Step.key(focus_team), Step.wait(60), Step.key(focus_team),
                       Step.move(ctx.park)], commands=1)
            moved = yield Poll(lambda s, v: camera_moved(world.radar.value, before), 0.7, 'camera on the army')
        view = world.radar.value
        at_front = front is not None and view is not None and view.online and view.in_view(front)
        rounds = max(1, int(seconds / every)) if (moved or at_front) and front is not None else (1 if front is None else 0)
        if prism_team and front is None:
            # The first centered view confirms safety; keep a short window for
            # the verified siege group to act before returning the camera home.
            rounds = min(3, max(1, int(seconds / every)))
        for index in range(rounds):
            if index and front is None and prism_team is None:
                break
            yield Poll(lambda s, v: False, 0.12 if index == 0 else every)    # render; then watch the orders work
            if index and leave is not None and (yield Look(lambda s, v: leave(s, index))):
                break
            view = world.radar.value
            if front is None:
                # Missing radar dots and an unchanged camera do not prove the saved team is absent.
                ours = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                  if ctx.last_canvas is not None else [])
                if not ours and prism_team:
                    from .allied_tactics import select_role
                    ours = yield from select_role(ctx, prism_team)
                if not ours and prism_team:
                    # An empty siege subgroup does not establish that the whole army is gone.
                    campaign.prism_roles.pop('prism_tank', None)
                    campaign.prism_probe_at = -1e9
                    campaign.prism_probe_offset = 0
                    campaign.prism_clear_since = None
                    yield Act([Step.click(ctx.clear_point, 'right'), Step.key(TEAM), Step.wait(80),
                               Step.key(TEAM), Step.move(ctx.park)], commands=0)
                    yield Poll(lambda s, v: False, 0.2)
                    ours = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                      if ctx.last_canvas is not None else [])
                    prism_team = None
                seen_units.append(len(ours))
                if not ours:
                    break
                campaign.army_seen_at = world.clock()
                campaign.confirmed_front = view.view_center if view is not None and view.online else None
                campaign.front_seen_at = world.clock()
                campaign.loss_confirmed = False
            if index == 0 and intel is not None:
                seen = yield Look(lambda s, v: survey(ctx.last_canvas, tactical, compressed=getattr(ctx, 'compressed_frames', False))
                                  if ctx.last_canvas is not None else None)
                if seen is not None and view is not None and view.view is not None:
                    intel.record_look('army', view.view_center, seen)
            center = (army_center(ours) if front is None else
                      view.to_tactical(front, tactical) if view is not None and view.online and view.in_view(front)
                      else (tactical.x + tactical.w // 2, tactical.y + tactical.h // 2))
            if army is not None and army.prism_plan() is not None:
                from .allied_tactics import round_order
                orders.extend((yield from round_order(ctx, campaign, army.book, army.broker)))
                continue
            order = yield Look(lambda s, v: battle_order(ctx.last_canvas, tactical, center,
                                                        compressed=getattr(ctx, 'compressed_frames', False)))
            if order is None:
                if index:
                    break                                    # nothing left here: the waves re-target
                continue
            if front is not None:
                # Select the team (nothing selected before 1: pressed on a selected team it could center
                # the camera); its health bars show where our units stand. A double tap has done it already.
                yield Act([Step.click(ctx.clear_point, 'right'), Step.key(TEAM), Step.move(ctx.park)], commands=0)
                yield Poll(lambda s, v: False, 0.15)
                ours = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                  if ctx.last_canvas is not None else [])
                seen_units.append(len(ours))
            if not ours:
                break                       # a cached view cannot establish where the selected team is now
            if view is not None and view.online:
                campaign.confirmed_front, campaign.front_seen_at = view.view_center, world.clock()
            center = army_center(ours)
            order = yield Look(lambda s, v: battle_order(ctx.last_canvas, tactical, center,
                                                        compressed=getattr(ctx, 'compressed_frames', False)))
            if order is None:
                break                                        # the enemy left while selection was rendered
            kind, point = order
            yield Act([Step.click(point, modifiers=ATTACK_MOVE if kind == 'fight' else FORCE_ATTACK),
                       Step.move(ctx.park)], commands=1)
            orders.append(order)
        if not stay:
            yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
            home = True
            yield Poll(lambda s, v: False, 0.2)
    finally:
        ctx.camera = 'home' if home else 'away' if stay else 'lost'
    if front is None:
        if moved or any(seen_units):
            campaign.focus_misses = 0
        elif radar is None or not radar.online:
            campaign.focus_misses += 1
            if campaign.focus_misses >= FOCUS_MISSES:
                campaign.teams = False
    world.note('order', order='focus', camera=bool(moved), fight=orders[-1][0] if orders else None,
               at=list(orders[-1][1]) if orders else None, orders=[o[0] for o in orders], teams=campaign.teams,
               front=list(front) if front else None, units_seen=seen_units)


def census(ctx, army):
    """Count the gathering army, or home reinforcements during an assault. After regrouping,
    center on the saved army; otherwise use H. P selects the units on screen, counted by their health
    bars (the frame says infantry or vehicle, whatever the damage: perception/unitboxes.py),
    deselect. The produced count cannot tell who is alive (live games 26 and 27: "12 tanks" attacked
    with the survivors of the rush and died in a minute).

    While census pairs are recorded for the unit detector (ctx.pairs, tools/detector.py), the views
    just before P and just after the deselect are kept with the selected one: the same units,
    unselected, labeled by their health bars a moment later or earlier."""
    pairs = ctx.pairs
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h')], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    focused = army.regrouped_defense is not None and not army.campaign.active
    home_units = None
    if focused:
        yield Act([Step.key('p'), Step.wait(150), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.12)
        home_units = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                if ctx.last_canvas is not None else None)
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        ctx.camera = 'away'
        yield Act([Step.key(TEAM), Step.wait(80), Step.key(TEAM), Step.wait(180), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.2)
    before = (yield Look(lambda s, v: ctx.frame_copy())) if pairs else None
    yield Act([Step.key('p'), Step.wait(150), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.12)
    selected = (yield Look(lambda s, v: ctx.frame_copy())) if pairs else None
    units = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                       if ctx.last_canvas is not None else None)
    yield Act([Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=0)
    if home_units is not None:
        def weight(view):
            return sum(1 if u.kind == 'vehicle' else army.book['army']['counted_infantry_weight']
                       for u in view or [])
        # Either view can cut off reinforcements. Do not add overlapping observations.
        if weight(home_units) > weight(units):
            units = home_units
    army.record_census(units)
    if pairs:
        yield Poll(lambda s, v: False, 0.12)
        after = yield Look(lambda s, v: ctx.frame_copy())
        pairs(before, selected, after)
    if focused:
        yield Act([Step.key('h'), Step.move(ctx.park)], commands=0)
        yield Poll(lambda s, v: False, 0.2)
        ctx.camera = 'home'


def look(ctx, radar_point, kind, intel):
    """Move only the camera to a radar point, survey the view for the battlefield picture, return."""
    tactical = ctx.layout.tactical

    def record(canvas, radar):
        seen = survey(canvas, tactical, compressed=getattr(ctx, 'compressed_frames', False)) if canvas is not None else None
        if seen is not None:
            intel.record_look(kind, radar_point, seen)
        return seen
    yield from attack_wave(ctx, radar_point, kind, order='look', on_view=record)


def base_watch(ctx, army, point, rounds=None):
    """Reveal an explored approach, fight any visible raiders, then return home."""
    home = False
    ctx.camera = 'away'
    try:
        moved = yield from radar_jump(ctx, point)
        if moved:
            yield Poll(lambda s, v: army._leave_battle(s), 0.15)
            if army.intel is not None:
                seen = yield Look(lambda s, v: survey(ctx.last_canvas, ctx.layout.tactical,
                                  compressed=ctx.compressed_frames))
                army.intel.record_look('base', point, seen)
            yield Look(lambda s, v: miners.observe_field(ctx, army.miners,
                       army.book['resources'], army.clock()))
            contact = yield Look(lambda s, v: army._watch_contact())
            if contact is not None:
                radar = ctx.world.radar.value
                contact_view = radar is not None and radar.in_view(contact)
                if not contact_view:
                    contact_view = yield from radar_jump(ctx, contact)
                if contact_view:
                    # A radar contact can be on a different approach from the one being inspected.
                    plan = army.book['defense']
                    yield from defense_session(ctx, plan['watch_rounds'] if rounds is None else rounds,
                                               plan['order_every'], plan['reach'], home=False,
                                               keep_home_infantry=army.book['attack'].get('keep_home_infantry', False))
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
        home = True
        yield Poll(lambda s, v: False, 0.2)
    finally:
        ctx.camera = 'home' if home else 'lost'
    ctx.world.note('base_watch', target=list(point), camera=bool(moved))


def event_watch(ctx, army):
    """Inspect a recent EVA incident before radar is available; fresh pixels decide the response."""
    home = False
    keep = army.book['attack'].get('keep_home_infantry', False)
    formed = False
    try:
        if army.unit_count():
            formed = (yield from field_team(ctx, allow_remote=True)) if keep else True
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key('space'), Step.move(ctx.park)], commands=0)
        ctx.camera = 'away'
        yield Poll(lambda s, v: False, 0.2)
        seen = yield Look(lambda s, v: survey(ctx.last_canvas, ctx.layout.tactical,
                          compressed=ctx.compressed_frames))
        if army.intel is not None:
            army.intel.record_look('base', (), seen)
        target = yield Look(lambda s, v: raider_target(ctx.last_canvas, ctx.layout.tactical,
                            ctx.layout.tactical.center, army.book['defense']['reach'],
                            compressed=ctx.compressed_frames))
        if target is not None:
            army.threat_at = army.watch_contact_at = army.clock()
            ctx.world.note('attack', near=list(target), source='event_watch')
            if formed:
                plan = army.book['defense']
                yield from defense_session(ctx, plan['session_rounds'], plan['order_every'],
                                           plan['reach'], home=False, keep_home_infantry=keep,
                                           team_formed=True)
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
        home = True
        yield Poll(lambda s, v: False, 0.2)
    finally:
        ctx.camera = 'home' if home else 'lost'
    ctx.world.note('event_watch', enemy_seen=target is not None)


def camera_home(ctx):
    """Press H after an interrupted order left the camera away from the base."""
    yield Act([Step.key('h')], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'


def resource_survey(ctx, army, point):
    """Inspect nearby explored ground for crystals and return the camera home."""
    home, moved = False, False
    ctx.camera = 'away'
    try:
        moved = yield from radar_jump(ctx, point)
        if moved:
            yield Poll(lambda s, v: army._leave_battle(s), 0.15)
            yield Look(lambda s, v: miners.observe_field(ctx, army.miners,
                       army.book['resources'], army.clock()))
        yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
        home = True
        yield Poll(lambda s, v: False, 0.2)
    finally:
        ctx.camera = 'home' if home else 'lost'
    ctx.world.note('resource_survey', target=list(point), camera=moved,
                   found=miners.field_fresh(army.miners, army.book['resources'], army.clock()))


def return_home(ctx, team=None):
    """Bring a saved team, or otherwise the whole army, back beside the base."""
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h')], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    select = SELECT_ALL if team is None else [Step.key(team), Step.wait(150)]
    yield Act(select + [Step.click(ctx.front_point())], commands=2)
    yield Act([Step.click(ctx.clear_point, 'right'), Step.move(ctx.park)], commands=0)
    ctx.world.note('order', order='return_home', at=list(ctx.front_point()))


def regroup_defenders(ctx):
    """Recall map-wide military after defense gives way to gathering at home."""
    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
    yield Poll(lambda s, v: False, 0.25)
    ctx.camera = 'home'
    point = ctx.front_point(reach=350)
    yield Act([Step.click(ctx.clear_point, 'right')] + SELECT_ALL +
              [Step.chord('ctrl', TEAM), Step.click(point, modifiers=ATTACK_MOVE), Step.move(ctx.park)], commands=1)
    yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
    ctx.world.note('order', order='regroup', at=list(point), arrived=False)


def clear_factory_exit(ctx):
    """Move verified combat vehicles away from an exit with a finished vehicle waiting."""
    if (yield from field_team(ctx)):
        units = our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
        occupied = [(u.bar[0] + u.bar[2] // 2, u.bar[1] + 25) for u in units]
        ground = ground_mask(ctx.last_canvas)
        points = open_spots(ctx.last_canvas, ctx.home, avoid=occupied,
                            distances=(240, 300, 360), minimum=.9, keep_out=80)
        # A contextual move onto a friendly vehicle guards it rather than clearing the exit.
        point = next((p for p in points if ground[p[1] - 6:p[1] + 7, p[0] - 6:p[0] + 7].mean() >= .9), None)
        if point is None:
            yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
            return
        yield Act([Step.key(TEAM), Step.wait(150), Step.click(point, modifiers=('ctrl', 'shift')),
                   Step.move(ctx.park)], commands=1)
        yield Act([Step.click(ctx.clear_point, 'right')], commands=0)
        ctx.world.note('order', order='clear_factory_exit', at=list(point))


class Campaign:
    """The attack in progress: current target, waves sent, guesses found empty."""

    def __init__(self):
        self.active = False
        self.started = None
        self.target = None
        self.kind = None
        self.guess_since = None
        self.waves = 0
        self.last_wave = -1e9
        self.cleared = []
        self.army_seen_at = None         # when our dots were last near the target
        self.verified_at = None          # a positive view, rather than radar contact alone
        self.sieging = False             # the last wave attacked a building our units stood beside
        self.teams = True                # waves save the army as team 1; focus by pressing 1 twice
        self.units_at_merge = 0          # army size when new units last joined team 1
        self.team_formed = False         # a wave has saved team 1
        self.last_focus = -1e9
        self.focus_misses = 0
        self.confirmed_front = None      # fresh team view when tiny radar dots disappear
        self.front_seen_at = None       # this camera position's own observation time, not general radar contact
        self.loss_confirmed = False
        self.depleted_confirmed = False
        self.prism_roles = {}
        self.prism_role_wave = -1
        self.prism_probe_at = -1e9
        self.prism_probe_offset = 0
        self.prism_clear_since = None
        self.prism_view = None
        self.prism_travel = None
        self.prism_move = None
        self.prism_staging_failed = False


class Army:
    """Checks about once a second; guards, defends, or sends attack waves (see module doc)."""

    def __init__(self, ctx, runner=None, intent=None, clock=time.monotonic, log=None, every=1.0, intel=None,
                 book=None):
        self.ctx, self.runner, self.intent, self.clock, self.log = ctx, runner, intent, clock, log
        self.intel = intel
        self.every = every                # seconds between checks
        self.book = book or playbook.default()
        self.broker = None
        self.last_check = self.last_order = self.last_note = -1e9
        self.regrouped_defense = None
        self.factory_ready_since = None
        self.factory_clear_at = -1e9
        self.seen = None                 # latest threat in view (point, pixels)
        self._small_threat = None        # (point, sample time), only for confirming weak JPEG contacts
        self.home_threat = None          # latest enemy group near home on the radar (point, dots)
        self.threat_at = -1e9            # when a threat near the base was last seen
        self.committed = 0               # combat units sent in earlier campaigns (not counted as available)
        self.failures = 0                # attacks whose army was destroyed (a fact for the attack size question)
        self.campaign = Campaign()
        self.census_tanks = None         # combat units counted at home (vehicles; infantry weighed by the profile)
        self.census_at = -1e9
        self.required_size = None        # callable(now) -> tanks needed for an attack (System One's attack size)
        self.census_enabled = True
        self.scouts_ready = False        # the dog scouts are saved as their own team
        self.explored = False            # they walked the loop around the base
        self.exploring = False           # ... and are walking it now
        self.dogs_at = None              # when the scouts were all built (other infantry waits a while)
        self.scout_tries = 0
        self.last_scout_try = -1e9
        self.census_healthy_vehicles = 0
        self.census_healthy_infantry = 0
        self.patrol = patrol.PatrolState()
        self.miners = miners.MinerGuard()
        self.patrol_checked = -1e9
        self.patrol_verified = False
        self.patrol_health = None
        self.explore_failed = False
        self.explore_checked = -1e9
        self.explore_deadline = 0
        self.explore_departed = False
        self.explore_home = None
        self.explore_origin = ()
        self.explore_count = 0
        self.watch_at = self.clock()
        self.watch_direction = 0
        self.watch_contact_at = -1e9
        self.watched_event = None

    # -- facts for decisions ------------------------------------------------
    def army_size(self):
        return combat_units(self.ctx.world, self.book)

    def unit_count(self):
        return combat_count(self.ctx.world, self.book)

    def available(self):
        """Combat units gathering at home: the last count of them when recent, else the ones
        produced and not yet committed to an earlier attack (losses are not seen there)."""
        if self.census_tanks is not None and self.clock() - self.census_at <= self.book['army']['census']['fresh_within']:
            return int(self.census_tanks)
        return max(0, self.army_size() - self.committed)

    def combat_built(self):
        """Combat unit types produced so far, with counts (the candidates for a unit's name)."""
        return {item: n for item, n in self.ctx.world.units_built.items()
                if n and item not in self.book['army']['not_combat']}

    def record_census(self, units):
        if units is None:
            return
        vehicles = sum(1 for u in units if u.kind == 'vehicle')
        infantry = sum(1 for u in units if u.kind == 'infantry')
        self.census_tanks = vehicles + self.book['army']['counted_infantry_weight'] * infantry
        self.census_at = self.clock()
        self.census_healthy_vehicles = sum(u.kind == 'vehicle' and u.health is not None and u.health >= self.book['patrol']['min_health']
                                           for u in units)
        self.census_healthy_infantry = sum(u.kind == 'infantry' and u.health is not None and u.health >= self.book['patrol']['min_health']
                                           for u in units)
        named = recognize(units, self.combat_built(), self.ctx.world.knowledge['items'])
        health = [u.health for u in units if u.health is not None]
        self.ctx.world.note('census', vehicles=vehicles, infantry=infantry,
                            names=dict(Counter(u.name or u.kind for u in named)),
                            health=round(sum(health) / len(health), 2) if health else None)

    def holding_for_scouts(self, now):
        """Other infantry waits until the dogs are grouped (or for a while at most)."""
        plan = self.book['units']['scouts']
        return (not self.scouts_ready and self.dogs_at is not None and self.scout_tries < 6 and
                now - self.dogs_at < plan['hold_others'])

    def _scouts_due(self, now):
        plan = self.book['units']['scouts']
        if self.ctx.world.units_built.get(plan['item'], 0) < plan['count'] or self.scouts_ready:
            return False
        if self.dogs_at is None:
            self.dogs_at = now
        return self.scout_tries < 6 and now - self.last_scout_try >= 3.0 and not self._recent_threat(now)

    def _census_due(self, now, every=None):
        if every is None:
            plan = self.book['army']['census']
            every = plan.get('during_attack', plan['every']) if self.campaign.active else plan['every']
        return self.census_enabled and now - self.census_at >= every and not self._recent_threat(now)

    def _recent_threat(self, now=None):
        """Keep defense priority while another task yields, without remembering an attack point."""
        now = self.clock() if now is None else now
        return self.ctx.world.mode.value == 'hud' and (self.seen is not None or self.home_threat is not None or
                                                      0 <= now - self.threat_at <= 3.0)

    def posture(self):
        return self.intent.posture if self.intent else 'gather'

    def threat_age(self, now=None):
        return (now or self.clock()) - self.threat_at

    def status(self, now=None):
        """Plain facts about the army (the reflex words them for System One)."""
        now = now or self.clock()
        return dict(posture=self.posture(), size=self.army_size(), available=self.available(),
                    attacking=self.campaign.active, target=self.campaign.target, kind=self.campaign.kind,
                    waves=self.campaign.waves, failures=self.failures, threat_age=round(self.threat_age(now), 1),
                    patrol=self.patrol.active, patrol_composition=self.patrol.composition,
                    exploring=self.exploring, explore_failed=self.explore_failed)

    def raid(self):
        """'big' or 'small' for the threat near the base now (None without one): a big raid is
        worth interrupting an attack for (the profile's defense section)."""
        if self.seen is None and self.home_threat is None:
            return None
        plan = self.book['defense']
        big = ((self.seen is not None and self.seen[1] >= plan['big_raid_pixels']) or
               (self.home_threat is not None and self.home_threat[1] >= plan['big_raid_dots']))
        return 'big' if big else 'small'

    # -- per tick -------------------------------------------------------------
    def _blocked(self, ignore_army=False):
        """Army orders wait while a sidebar task owns the camera and cursor, or the cursor is unknown."""
        if self.runner is None or (self.runner.busy('army') and not ignore_army):
            return True
        sidebar = self.runner.tasks.get('sidebar')
        if sidebar is not None and not sidebar.name.startswith('look:'):
            return True
        pointer = self.ctx.pointer
        return pointer is not None and getattr(pointer, 'name', '') == 'locked' and pointer.position is None

    def _look(self, now):
        """Threats near the base: the tactical view (while the camera is home) and the radar."""
        world, ctx = self.ctx.world, self.ctx
        if world.mode.value != 'hud':
            self.seen = self.home_threat = None      # menus and score screens have blue parts too
            self._small_threat = None
            return
        miners.observe_home(ctx, self.miners, self.book['resources'], now)
        compressed = getattr(ctx, 'compressed_frames', False)
        sidebar = self.runner.tasks.get('sidebar') if self.runner is not None else None
        placing = sidebar is not None and sidebar.name.startswith('place:')
        # Our red placement preview is not an enemy. Radar contacts remain available below.
        self.seen = threat(ctx.last_canvas, compressed=compressed) if ctx.camera == 'home' and not placing else None
        if compressed and ctx.camera == 'home' and self.seen is None and not placing:
            small = threat(ctx.last_canvas, min_blob=12, min_total=24, compressed=True)
            if small is None:
                tactical = ctx.layout.tactical
                groups = _compressed_raider_groups(ctx.last_canvas, tactical, tactical.center,
                                                    tactical.w + tactical.h)
                if groups:
                    area, point = max(groups, key=lambda group: group[0])
                    small = point, area
                else:
                    point = _standing_raider(ctx.last_canvas, tactical, tactical.center,
                                             tactical.w + tactical.h)
                    if point is not None:
                        small = point, 0              # appearance evidence, not a fabricated pixel count
            previous = self._small_threat
            if previous is not None and now - previous[1] > 3.0:
                previous = None
            nearby = (small is not None and previous is not None and
                      sum(abs(a - b) for a, b in zip(previous[0], small[0])) <= 90)
            # Keep the first sample time; fast captures must not postpone confirmation forever.
            self._small_threat = previous if small is None or nearby else (small[0], now)
            # JPEG can leave only a few pixels on a real raider. Two nearby observations
            # establish a threat despite brief missing frames. Missing frames cannot confirm it.
            if nearby:
                sampled = previous[1]
                if 0.5 <= now - sampled <= 3.0:
                    self.seen = small                  # current evidence, never the remembered point
        else:
            self._small_threat = None
        radar, home = world.radar.value, ctx.radar_home
        self.home_threat = None
        if radar is not None and radar.online and home is not None:
            near = radar.enemy_near(home, HOME_RADIUS + (self.book['patrol']['threat_radius_extra'] if self.patrol.active else 0))
            if near:
                self.home_threat = max(near)[1], sum(a for a, _ in near)
        if self.seen is not None or self.home_threat is not None:
            self.threat_at = now
            if now - self.last_note >= 5.0:
                self.last_note = now
                if self.seen is not None:
                    world.note('attack', near=list(self.seen[0]), pixels=self.seen[1])
                else:
                    world.note('attack', radar=list(self.home_threat[0]), dots=self.home_threat[1])

    def step(self, snapshot, now=None, act=True):
        """Returns a short description of the intended order (or None)."""
        now = now or self.clock()
        if now - self.last_check < self.every:
            return None
        self.last_check = now
        if self.ctx.camera == 'lost' and act and not self._blocked():
            self.runner.start('army:camera', 'army', camera_home(self.ctx), priority=2)
            return ('army:camera', None)
        self._look(now)
        current = self.runner.tasks.get('army') if self.runner is not None else None
        if (current is not None and current.name == 'army:attack' and
                ((self.ctx.camera == 'home' and self.seen is not None) or
                 (not self.campaign.waves and self.home_threat is not None))):
            # Home role formation can take many frames; a fresh raid must not wait for it.
            point = self.seen[0] if self.ctx.camera == 'home' and self.seen is not None else self.home_threat[0]
            if act and not self._blocked(ignore_army=True):
                task = (self._defense() if self.ctx.camera == 'home' and self.seen is not None else
                        base_watch(self.ctx, self, point, rounds=self.book['defense']['session_rounds']))
                if self.runner.start('army:defend', 'army', task, priority=1, preempt=True):
                    self.last_order = now
            return ('army:defend', point)
        miners.monitor_income(self.ctx, self.miners, self.book['resources'], now)
        if self.intel is not None:
            self.intel.update(self.ctx.world.radar.value, self.ctx.radar_home, now)
        if self.intent is not None and self.intent.focus == 'attack' and self.posture() == 'attack':
            return self._attack(now, act)      # optional surveys and scouting wait during the assault
        radar = self.ctx.world.radar.value
        world = self.ctx.world
        event = next((e for e in reversed(world.events)
                      if e['kind'] in ('attack', 'structure-lost') and e.get('source') != 'event_watch'
                      and 0 <= now - world.started - e['t'] <= 20), None)
        token = (event['t'], event['kind'], event.get('item')) if event is not None else None
        if ((radar is None or not radar.online) and event is not None and token != self.watched_event
                and (event['kind'] == 'structure-lost' or self.seen is None)
                and now - self.watch_at >= self.book['defense']['watch_every']
                and not any(snapshot.get('lanes', {}).get(lane, {}).get('status') == 'ready'
                            for lane in ('buildings', 'defenses'))):
            if act and not self._blocked():
                if self.runner.start('army:event-watch', 'army', event_watch(self.ctx, self), priority=2):
                    self.watch_at, self.watched_event = now, token
            return ('army:event-watch', None)
        if (self.home_threat is not None and radar is not None and
                not radar.in_view(self.home_threat[0])):
            point = self.home_threat[0]
            if act and not self._blocked() and now - self.last_order >= self.book['defense']['repeat']:
                if self.runner.start('army:defend', 'army', base_watch(
                        self.ctx, self, point, rounds=self.book['defense']['session_rounds']), priority=2):
                    self.last_order = now
            return ('army:defend-radar', point)
        if self.posture() == 'gather' and not self.campaign.active and not self._recent_threat(now):
            defense = next((e for e in reversed(world.events)
                            if e['kind'] == 'order' and e.get('order') == 'defense' and e.get('orders', 0)), None)
            if defense is not None and defense['t'] != self.regrouped_defense:
                if act and not self._blocked():
                    if self.runner.start('army:regroup', 'army', regroup_defenders(self.ctx), priority=2):
                        self.regrouped_defense = defense['t']
                return ('army:regroup', None)
        miner_plan = self.book['defense']['emergency_miners']
        miner_group = miners.can_group(self.ctx.world, self.miners, miner_plan)
        if miners.can_recover_group(self, now):
            if act and not self._blocked():
                self.runner.start('army:miner-recover', 'army',
                                  miners.recover_group(self.ctx, self.miners, miner_plan), priority=2)
            return ('army:miner-recover', None)
        if miner_group or miners.due(self, now):
            name = 'army:miner-group' if miner_group else 'army:miner-support'
            overdue = self.miners.engaged and now - self.miners.started >= miner_plan['max_seconds']
            current = self.runner.tasks.get('army') if self.runner is not None else None
            preempt = overdue and current is not None and current.name != name
            if act and not self._blocked(ignore_army=preempt):
                task = (miners.group(self.ctx, self.miners, miner_plan) if miner_group else
                        miners.support(self.ctx, self.miners, miner_plan, self.clock))
                self.runner.start(name, 'army', task, priority=1 if preempt else 2, preempt=preempt)
            return (name, None)
        if miners.survey_due(self, now):
            point, name = miners.survey_point(self), 'army:resource-survey'
            if act and not self._blocked():
                if self.runner.start(name, 'army', resource_survey(self.ctx, self, point), priority=3):
                    self.miners.survey_at = now
                    self.miners.survey_direction += 1
            return (name, point)
        if miners.resource_due(self, now):
            name = 'army:resource'
            if act and not self._blocked():
                self.runner.start(name, 'army', miners.resource_order(
                    self.ctx, self.miners, self.book['resources'], self.clock,
                    resource='ore' if self.intent.resources == 'harvest_home_ore' else 'gems'), priority=3)
            return (name, self.miners.field)
        if self.exploring and (now >= self.explore_deadline or self._recent_threat(now)):
            self.exploring, self.explore_failed = False, True
            self.ctx.world.note('explore', arrived=False, reason='threat' if self._recent_threat(now) else 'timeout')
        posture = self.posture()
        if posture == 'attack':
            plan = self._attack(now, act)
            return self._intel(now, act) or plan      # looks wait while a wave is under way
        if self.campaign.active and posture == 'gather':
            self._end_campaign(now, 'the army was called home')
        repeat = self.book['defense']['repeat']
        if (self.seen is not None or (self.home_threat is None and self._recent_threat(now))) and self.unit_count():
            plan = ('army:defend', self.seen[0] if self.seen is not None else None)
            if act and not self._blocked() and now - self.last_order >= repeat:
                if self.runner.start('army:defend', 'army', self._defense(), priority=2):
                    self.last_order = now
            return plan
        ground = set(self.book['units']['tanks']) | {self.book['units']['anti_air']['item'],
                                                   self.book['units']['siege']['item'],
                                                   self.book['units']['miners']['item']}
        slots = world.slots['vehicles']
        ready = (slots.fresh(now, 2) and
                 any(s.item in ground and s.state == 'ready' for s in slots.value or []))
        if slots.fresh(now, 2) and not ready:
            self.factory_ready_since = None
        elif ready and self.factory_ready_since is None:
            self.factory_ready_since = now
        elif (ready and not self.campaign.active and self.ctx.camera == 'home' and not self._recent_threat(now)
              and now - self.factory_ready_since >= 3 and now - self.factory_clear_at >= 20):
            if act and not self._blocked():
                if self.runner.start('army:clear-exit', 'army', clear_factory_exit(self.ctx), priority=2):
                    self.factory_clear_at = now
            return ('army:clear-exit', None)
        if self._scouts_due(now):
            if act and not self._blocked():
                team = self.book['intel']['scout_team']
                count = self.book['units']['scouts']['count']
                if self.runner.start('army:scouts', 'army', group_scouts(self.ctx, self, team, count), priority=3):
                    self.scout_tries += 1
                    self.last_scout_try = now
            return ('army:scouts', None)
        if self.unit_count() and self._census_due(now):
            if act and not self._blocked():
                self.runner.start('army:census', 'army', census(self.ctx, self), priority=3)
            return ('army:census', None)
        if (posture == 'defend' or self.patrol.active or 0 <= now - self.watch_contact_at <= 3) and self.home_threat is not None and self.unit_count():
            point = self.home_threat[0]
            plan = ('army:defend-radar', point)
            if act and not self._blocked() and now - self.last_order >= repeat:
                radar = self.ctx.world.radar.value
                task = (base_watch(self.ctx, self, point, rounds=self.book['defense']['session_rounds'])
                        if radar is not None and not radar.in_view(point) else
                        attack_wave(self.ctx, point, 'defend',
                                    keep_home_infantry=self.book['attack'].get('keep_home_infantry', False)))
                if self.runner.start('army:defend', 'army', task, priority=2):
                    self.last_order = now
            return plan
        if not self.campaign.active and not self._recent_threat(now) and 'waypoint' in self.ctx.layout.points:
            task, name = None, None
            plan = self.book['patrol']
            if (now - self.census_at <= self.book['army']['census']['fresh_within'] and
                    self.census_healthy_infantry >= plan['infantry'] and
                    patrol.can_form(self.ctx.world, self.patrol, self.census_healthy_vehicles, self.threat_age(now), self.book)):
                name, task = 'patrol', patrol.form_patrol(self.ctx, self.patrol,
                                                        patrol.guard_route(self.ctx, book=self.book), self.book)
            elif self.patrol.active and now - self.patrol_checked >= plan['check_every']:
                name, task = 'patrol-check', monitor_patrol(self.ctx, self)
            elif self.exploring and now - self.explore_checked >= self.book['intel']['explore']['check_every']:
                name, task = 'explore-check', monitor_exploration(self.ctx, self)
            if task is not None:
                if act and not self._blocked():
                    self.runner.start(f'army:{name}', 'army', task, priority=3)
                return (f'army:{name}', None)
        intel_plan = self._intel(now, act)
        if intel_plan is not None:
            return intel_plan
        if (posture in ('gather', 'defend') and not self.campaign.active and now - self.watch_at >= self.book['defense']['watch_every'] and
                self.ctx.camera == 'home' and not self._leave_battle(snapshot)):
            point = self._watch_point()
            if point is not None:
                if act and not self._blocked():
                    if self.runner.start('army:base-watch', 'army', base_watch(self.ctx, self, point), priority=4):
                        self.watch_at, self.watch_direction = now, self.watch_direction + 1
                return ('army:base-watch', point)
        return None

    def _watch_point(self):
        radar, home = self.ctx.world.radar.value, self.ctx.radar_home
        if radar is None or not radar.online or radar.view is None or home is None:
            return None
        x0, y0, x1, y1 = radar.view
        rx, ry = min(HOME_RADIUS, max(4, (x1 - x0) // 2)), min(HOME_RADIUS, max(4, (y1 - y0) // 2))
        offsets = ((0, -ry), (rx, 0), (0, ry), (-rx, 0))
        known = self.intel.enemy_base if self.intel is not None else None
        # Large own radar contacts can be buildings beyond the home viewport. Inspect
        # them, but never change building counts from a dot's size or an unseen object.
        points = [point for area, point in radar.own if area >= BASE_DOTS and not radar.in_view(point)
                  and radar.explored_near(point, radius=2) >= 0.6]
        for dx, dy in offsets:
            point = radar.inside((home[0] + dx, home[1] + dy))
            if (3 <= distance(point, home) <= HOME_RADIUS and radar.explored_near(point, radius=2) >= 0.6 and
                    (known is None or distance(point, known) > HOME_RADIUS) and point not in points):
                points.append(point)
        return points[self.watch_direction % len(points)] if points else None

    def _watch_contact(self):
        """Current visible pixels or radar dots establish contact, never the chosen camera target."""
        ctx, world = self.ctx, self.ctx.world
        radar, home = world.radar.value, ctx.radar_home
        if radar is None or not radar.online or radar.view is None or home is None:
            return None
        near = radar.enemy_near(home, HOME_RADIUS)
        contact = max(near)[1] if near else None
        seen = threat(ctx.last_canvas, compressed=ctx.compressed_frames)
        if contact is None and seen is not None:
            tactical = ctx.layout.tactical
            x0, y0, x1, y1 = radar.view
            px, py = seen[0]
            point = (round(x0 + (px - tactical.x) / tactical.w * (x1 - x0)),
                     round(y0 + (py - tactical.y) / tactical.h * (y1 - y0)))
            if tactical.contains(seen[0]) and distance(point, home) <= HOME_RADIUS:
                contact = point
        if contact is not None:
            self.threat_at = self.watch_contact_at = self.clock()
            world.note('attack', radar=list(contact), source='base_watch',
                       pixels=seen[1] if seen is not None else 0, dots=sum(a for a, _ in near))
        return contact

    def _scout_point(self, now):
        """Nearby shroud in the home viewport, away from known enemies and their base."""
        world, home = self.ctx.world, self.ctx.radar_home
        plan = self.book['patrol']
        scout = plan['scout']
        if (not self.patrol_verified or now - self.patrol_checked > scout['fresh_within'] or
                now - self.census_at > self.book['army']['census']['fresh_within'] or home is None or
                not patrol.can_scout(world, self.patrol, self.census_healthy_vehicles - plan['vehicles'],
                                     self.threat_age(now), self.patrol_health, self.book)):
            return None
        radar = world.radar.value
        if radar.view is None:
            return None
        x0, y0, x1, y1 = radar.view
        if x1 - x0 <= 4 or y1 - y0 <= 4:
            return None
        known = self.intel.enemy_base if self.intel is not None else None
        for dx, dy in scout['offsets']:
            # A home viewport can be narrower than 18 radar pixels. Aim just inside its edge.
            point = (min(max(home[0] + dx, x0 + 2), x1 - 2),
                     min(max(home[1] + dy, y0 + 2), y1 - 2))
            if (scout['min_distance'] <= distance(point, home) <= scout['max_distance'] and
                    radar.inside(point) == point and radar.in_view(point, margin=-2) and
                    not radar.enemy_near(point, HOME_RADIUS) and
                    (known is None or distance(point, known) > scout['enemy_base_distance']) and radar.explored_near(point) < 0.6):
                return point
        return None

    def _intel(self, now, act):
        """Carry out System One's intel request (scout, look), and bring a successful scout home."""
        intel, intent = self.intel, self.intent
        if intel is None or intent is None:
            return None
        radar, home = self.ctx.world.radar.value, self.ctx.radar_home
        team = self.book['intel']['scout_team']
        scout = intel.scout
        if (scout is not None and scout.get('found') and not scout.get('returned') and
                self.scouts_ready and self.posture() != 'attack'):
            if act and not self._blocked():
                if self.runner.start('army:return', 'army', return_home(self.ctx, team=team), priority=2):
                    scout['returned'] = True
            return ('army:return', None)
        request = intent.intel
        if request in (None, 'nothing'):
            return None
        if request == 'scout' and not self.scouts_ready:
            intent.set('intel', None, 'army: scout team unavailable')
            return None
        if request == 'scout' and self.exploring:
            return None                       # retain the request until verified return or route failure
        if request in ('explore', 'scout') and self._recent_threat(now):
            return None
        if (request == 'explore' and self.scouts_ready and not self.exploring and not self.explore_failed
                and 'waypoint' in self.ctx.layout.points):
            plan = self.book['intel']['explore']
            point, task = None, explore_route(self.ctx, self, team, plan['route'], plan['leg_seconds'])
        elif request == 'scout':
            point, kind = choose_target(radar, home, cleared=self.campaign.cleared)
            select = [Step.key(team), Step.wait(150)]
            task = (attack_wave(self.ctx, point, kind, order='move', select=select)
                    if point is not None else None)
        elif request == 'look_enemy_base' and intel.enemy_base is not None:
            point = tuple(intel.enemy_base)
            task = look(self.ctx, point, 'enemy_base', intel)
        elif request == 'look_at_army' and self.campaign.teams and self.campaign.team_formed:
            point, task = None, self._battle()
            self.campaign.last_focus = now
        elif request == 'look_at_army':
            point = (radar.army_center(home) if radar is not None and radar.online else None) or self.campaign.target
            task = look(self.ctx, point, 'army', intel) if point is not None else None
        else:
            point, task = None, None
        if task is None:
            intent.set('intel', None, 'army: nothing to do')
            return None
        plan = (f'army:{request}', point)
        if act and not self._blocked():
            if self.runner.start(f'army:{request}', 'army', task, priority=2):
                intent.set('intel', None, 'army: done')
                if request == 'scout':
                    intel.send_scout(point, now)
        return plan

    def _end_campaign(self, now, reason):
        self.committed = self.army_size()
        self.ctx.world.note('campaign', ended=reason, waves=self.campaign.waves)
        self.campaign = Campaign()

    def _army_lost(self, now, radar):
        """Radar suspects a loss; a fresh team view must confirm it before ending the attack.

        Live game 5: the first wave reached the enemy base and was destroyed; reinforcements then
        trickled in three at a time and died one group after another.
        """
        campaign = self.campaign
        if radar is None or not radar.online or campaign.target is None:
            return False
        if any(distance(point, campaign.target) <= FIELD_RADIUS for _, point in radar.own):
            campaign.army_seen_at = now
            campaign.loss_confirmed = False
            return False
        if (not campaign.loss_confirmed and campaign.verified_at is not None
                and campaign.last_focus < campaign.verified_at):
            # Let a verified live team get its combat turn before checking it again.
            return False
        since = campaign.army_seen_at if campaign.army_seen_at is not None else campaign.started
        return now - since >= (LOST_AFTER if campaign.army_seen_at is not None else 3 * LOST_AFTER) \
            and (campaign.army_seen_at is not None or now - campaign.last_wave >= LOST_AFTER)

    def _verify_army(self):
        """Missing radar dots require a fresh, established team view before declaring a loss."""
        ctx, campaign = self.ctx, self.campaign
        if not campaign.team_formed or not campaign.teams:
            campaign.loss_confirmed = False
            campaign.depleted_confirmed = False
            ctx.world.note('army_verified', units=[], lost=False, reason='team unavailable')
            return
        home = False
        ctx.camera = 'away'
        try:
            radar = ctx.world.radar.value
            at_front = False
            point = (front_point(radar, ctx.radar_home, campaign.target) or
                     (self.intel.enemy_base if self.intel is not None else None) or campaign.target)
            if radar is not None and radar.online and point is not None:
                moved = yield from radar_jump(ctx, point, 0.7)
                if not moved:
                    campaign.loss_confirmed = campaign.depleted_confirmed = False
                    ctx.world.note('army_verified', units=[], lost=False, reason='battle view unavailable')
                    yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
                    home = True
                    return
                at_front = True
            for view_index in range(2 if at_front else 1):
                if at_front and view_index == 0:
                    yield Act([Step.key(TEAM), Step.move(ctx.park)], commands=0)
                else:
                    # The army can still be traveling through an empty destination view.
                    yield Act([Step.click(ctx.clear_point, 'right'), Step.key(TEAM), Step.wait(100),
                               Step.key(TEAM), Step.move(ctx.park)], commands=0)
                counts, vehicles = [], []
                for _ in range(2):
                    yield Poll(lambda s, v: False, 0.3)
                    units = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                       if ctx.last_canvas is not None else None)
                    counts.append(None if units is None else len(units))
                    vehicles.append(None if units is None else sum(u.kind == 'vehicle' for u in units))
                if any(count for count in counts):
                    break
            # Assigning Allied type groups removes those units from group 1.
            # An empty general team cannot establish that its siege groups died.
            if not any(count for count in counts) and self.prism_plan() is not None:
                for team in campaign.prism_roles.values():
                    yield Act([Step.click(ctx.clear_point, 'right'), Step.key(team), Step.wait(100),
                               Step.key(team), Step.move(ctx.park)], commands=0)
                    counts, vehicles = [], []
                    for _ in range(2):
                        yield Poll(lambda s, v: False, 0.3)
                        units = yield Look(lambda s, v: our_units(ctx.last_canvas, compressed=ctx.compressed_frames)
                                           if ctx.last_canvas is not None else None)
                        counts.append(None if units is None else len(units))
                        vehicles.append(None if units is None else sum(u.kind == 'vehicle' for u in units))
                    if any(count for count in counts):
                        break
            # After a failed assault, a depleted team should regroup rather than draw another trickle.
            campaign.depleted_confirmed = (self.failures > 0 and
                                           not (self.prism_plan() is not None and campaign.prism_roles) and
                                           all(n is not None and 0 < n <= self.book['attack']['reinforce_units']
                                               for n in counts))
            if any(count for count in counts):
                campaign.army_seen_at = self.clock()
                campaign.verified_at = campaign.army_seen_at
                radar = ctx.world.radar.value
                campaign.confirmed_front = radar.view_center if radar is not None and radar.online else None
                campaign.front_seen_at = self.clock()
                campaign.loss_confirmed = False
            else:
                campaign.loss_confirmed = counts == [0, 0]
            ctx.world.note('army_verified', units=counts, vehicles=vehicles,
                           lost=campaign.loss_confirmed, depleted=campaign.depleted_confirmed)
            yield Act([Step.click(ctx.clear_point, 'right'), Step.key('h'), Step.move(ctx.park)], commands=0)
            home = True
            yield Poll(lambda s, v: False, 0.2)
        finally:
            ctx.camera = 'home' if home else 'lost'

    def _leave_battle(self, snapshot, rounds=99):
        """A battle session goes home early when enemies are at the base, or, after a few orders,
        when a finished building waits for placement (it blocks all construction; with money
        piling up one is almost always ready)."""
        lanes = snapshot.get('lanes', {})
        ready = any(lanes.get(lane, {}).get('status') == 'ready' for lane in ('buildings', 'defenses'))
        return (ready and rounds >= self.book['battle']['placement_after_orders']) or self._recent_threat()

    def _focus_due(self, now):
        campaign, plan = self.campaign, self.book['battle']
        return (campaign.teams and campaign.team_formed and now - campaign.last_focus >= plan['every'] and
                now - campaign.last_wave >= plan['after_wave'])

    def _defense(self):
        plan = self.book['defense']
        return defense_session(self.ctx, plan['session_rounds'], plan['order_every'], plan['reach'])

    def _battle(self):
        plan = self.book['battle']
        return battle_session(self.ctx, self.campaign, self.intel, self._leave_battle, plan['seconds'],
                              plan['order_every'], stay=self.intent is not None and self.intent.focus == 'attack', army=self)

    def prism_plan(self):
        return self.book['battle'].get('prism') if self.ctx.world.units_built.get('prism_tank') else None

    def _attack(self, now, act):
        world, campaign, plan = self.ctx.world, self.campaign, self.book['attack']
        if campaign.active and campaign.depleted_confirmed:
            self.failures += 1
            self._end_campaign(now, f'only isolated survivors remain ({self.failures} failed attacks)')
            if self.intent is not None:
                self.intent.set('posture', 'gather', 'rule: attack depleted')
                self.intent.set('attack_size', None, 'rule: attack depleted')
            return None
        if (campaign.active and self._army_lost(now, world.radar.value) and
                campaign.team_formed and campaign.teams):
            if not campaign.loss_confirmed:
                if act and not self._blocked():
                    self.runner.start('army:verify', 'army', self._verify_army(), priority=2)
                return ('army:verify', campaign.target)
            self.failures += 1
            self._end_campaign(now, f'the army was destroyed ({self.failures} failed attacks)')
            if self.intent is not None:
                self.intent.set('posture', 'gather', 'rule: attack failed')
                self.intent.set('attack_size', None, 'rule: attack failed')    # asked again, with the failure
            return None
        if self.seen is not None and not campaign.active:
            # Enemies inside the base while the army is still home: clear them before leaving.
            order = ('army:defend', self.seen[0])
            if act and not self._blocked() and now - self.last_order >= self.book['defense']['repeat']:
                if self.runner.start('army:defend', 'army', self._defense(), priority=2):
                    self.last_order = now
            return order
        if not campaign.active and self.census_enabled:
            if now - self.census_at > self.book['army']['census']['before_attack_within']:
                if act and not self._blocked():
                    self.runner.start('army:census', 'army', census(self.ctx, self), priority=2)
                return ('army:census', None)
            need = self.required_size(now) if self.required_size is not None else 0
            if self.census_tanks is not None and self.census_tanks < need:
                if self.intent is not None:
                    self.intent.set('posture', 'gather', f'rule: {self.census_tanks:.0f} tanks at home, {need} needed')
                return None
        # Slow input or a long formation can pass the entire between-wave window.
        # Inspect the sent army once before another census/wave takes the camera home.
        if (campaign.active and campaign.waves and
                (campaign.last_focus < campaign.last_wave or campaign.prism_staging_failed)
                and self._focus_due(now)):
            if act and not self._blocked():
                if self.runner.start('army:focus', 'army', self._battle(), priority=2):
                    campaign.last_focus = now
                    campaign.prism_staging_failed = False
            return ('army:focus', campaign.target)
        size = self.army_size()
        if campaign.active and self.census_enabled:
            if self._census_due(now):
                if act and not self._blocked():
                    self.runner.start('army:census', 'army', census(self.ctx, self), priority=2)
                return ('army:census', None)
            # Queued production does not establish how many reinforcements are ready at home.
            fresh = self.census_tanks or 0
        else:
            fresh = size - campaign.units_at_merge
        reinforce = not campaign.teams or not campaign.team_formed or fresh >= plan['reinforce_units']
        if self.prism_plan() is not None and campaign.active and (not reinforce or campaign.sieging):
            if act and self._focus_due(now) and not self._blocked():
                if self.runner.start('army:focus', 'army', self._battle(), priority=2):
                    campaign.last_focus = now
            return ('army:attack', campaign.target)
        every = plan['siege_every'] if campaign.sieging else plan['wave_every']
        due = not campaign.active or fresh >= plan['reinforce_units'] or now - campaign.last_wave >= every
        if not due or now - campaign.last_wave < min(plan['wave_gap'], every):
            if campaign.active and self._focus_due(now):
                if act and not self._blocked():
                    if self.runner.start('army:focus', 'army', self._battle(), priority=2):
                        campaign.last_focus = now
                return ('army:focus', None)
            return ('army:attack', campaign.target) if campaign.active else None
        timeout = plan.get('guess_timeout')
        since = campaign.guess_since if campaign.guess_since is not None else campaign.started
        if (timeout is not None and campaign.guess_since is not None and campaign.kind == 'guess' and campaign.target is not None and
                since is not None and now - since >= timeout and world.radar.value is not None and
                world.radar.value.explored_near(campaign.target) < 0.6 and
                campaign.target not in campaign.cleared):
            campaign.cleared.append(campaign.target)
            world.note('guess_abandoned', target=list(campaign.target), reason='still shrouded after timeout')
        target, kind = choose_target(world.radar.value, self.ctx.radar_home,
                                     campaign.target if campaign.kind == 'guess' else None, campaign.cleared,
                                     self.intel.enemy_base if self.intel is not None else None)
        if (campaign.kind == 'guess' and campaign.target is not None and target != campaign.target and
                campaign.target not in campaign.cleared):
            campaign.cleared.append(campaign.target)       # explored and empty: the enemy is elsewhere
        if target is None:
            if campaign.active and self._focus_due(now):
                if act and not self._blocked():
                    if self.runner.start('army:focus', 'army', self._battle(), priority=2):
                        campaign.last_focus = now
                return ('army:focus', campaign.target)
            return None
        order = ('army:attack', target)
        if not act or self._blocked():
            return order
        if self.runner.start('army:attack', 'army', attack_wave(self.ctx, target, kind, campaign=campaign,
                            reinforce=reinforce, keep_home_infantry=plan.get('keep_home_infantry', False),
                            prism=self.prism_plan()), priority=2):
            if reinforce:
                campaign.units_at_merge = size
                if self.census_enabled:
                    self.census_tanks = None
            if not campaign.active:
                campaign.active, campaign.started = True, now
                self.patrol.active = False       # the campaign merges all units into team 1
                if self.exploring:
                    self.exploring, self.explore_failed = False, True
                world.note('campaign', started=True, army=size, available=self.available())
            if target != campaign.target or kind != campaign.kind:
                campaign.guess_since = None    # Starts when the order is acknowledged, after team formation.
            campaign.target, campaign.kind = target, kind
            self.last_order = now
        return order
