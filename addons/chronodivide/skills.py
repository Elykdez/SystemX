"""Game-specific skills as task coroutines (see tasks.py), in canvas coordinates.

Every skill verifies its effect from later frames, mostly through the sidebar:
a queued item shows construction progress, a placed building leaves "Ready",
and a deployed MCV fills the Buildings tab. Nothing is repeated blindly.
"""

from collections import deque

from .executor import Step
from .perception.placement import read_ghost
from .perception.terrain import ground_mask, open_spots, openness
from .perception.units import base_extent, find_base, find_mcv
from .tasks import Act, Expect, Look, Poll, TaskFailed, Wait

HOME = (428, 360)          # construction yard / MCV at the tactical center (1024x768; layouts give 'home')
CELL = (60, 30)            # one map cell on screen (isometric diamond)


class Context:
    """Shared, read-mostly helpers for skills: layout geometry, world lookups, cursor vision."""

    def __init__(self, layout, world, cursor_finder=None, pointer=None, home=None, build_off_ally=False):
        self.world = world
        self.cursor_finder, self.pointer = cursor_finder, pointer
        self.use_layout(layout, home)
        self.camera = 'home'             # 'away' while an army order has moved the camera
        self.radar_home = None           # our base on the radar (canvas pixels), once the radar is online
        self.keyboard_tabs = True        # Q/W/E/R; disabled automatically if they stop working
        self.keyboard_tab_misses = 0
        self.gain_measured = False       # pointer gain is measured once; re-calibration only re-finds the cursor
        self.last_canvas = None
        self.compressed_frames = False  # browser screenshots may use JPEG chroma subsampling
        self.build_off_ally = build_off_ally
        self.pairs = None                # records census pairs for the unit detector (tools/detector.py)
        self.factory_exit = None        # reserved home-view access from a factory we placed

    def use_layout(self, layout, home=None):
        """Points of the game view's layout (again after the view changed size: home is found anew)."""
        self.layout = layout
        # An empty cameo slot at the bottom of the sidebar: the plain arrow is drawn there and no
        # tooltip opens (over the online radar the cursor changes shape with the selection).
        self.park = layout.points.get('park', (940, 690))
        # After a radar click the cursor rests on the radar: moving it straight off the radar dragged
        # the camera along to the map's edge (live games 29 and 30, park below the radar: every radar
        # click failed; games 15-28 parked here and none did).
        self.radar_park = layout.points.get('radar_park', (940, 103))
        self.clear_point = layout.points.get('recapture', (425, 650))   # a right click here clears the selection
        self.home = home or layout.points.get('home', HOME)   # construction yard on screen right after pressing H
        self.home_known = False

    def lane(self, item):
        return self.world.knowledge['items'][item]['lane']

    def tab_point(self, lane):
        return self.layout.tabs[lane].center

    def slot_point(self, index):
        return self.layout.grid.box(index).center

    def slot_index(self, item, states):
        lane = self.lane(item)
        if self.world.active_tab.value != lane:
            return None                           # cached slots from another tab are not clickable
        fact = self.world.slots[lane]
        for slot in fact.value or []:
            if slot.item == item and slot.state in states:
                return slot.index
        return None

    def placement_candidates(self, extent=None, ring=0, spacing=70, margin=60):
        """Probe points on rectangles around the base, nearest first, clear of the view's edges.

        Isometric ground: vertical distances are half the horizontal ones. Each rectangle is
        walked from its bottom side (open ground in front of the base) around to the top.
        """
        tactical = self.layout.tactical
        if extent is None:
            x, y = self.home
            extent = (x - 60, y - 40, x + 60, y + 40)
        x0, y0, x1, y1 = extent
        points = []
        for gap in (70, 130, 200, 280)[ring:]:
            left, right, top, bottom = x0 - gap, x1 + gap, y0 - gap // 2, y1 + gap // 2
            width, height = right - left, bottom - top
            for distance in range(0, 2 * (width + height), spacing):
                if distance < width:                                  # bottom, left to right
                    point = (left + distance, bottom)
                elif distance < width + height:                       # right side, upwards
                    point = (right, bottom - (distance - width))
                elif distance < 2 * width + height:                   # top, right to left
                    point = (right - (distance - width - height), top)
                else:                                                 # left side, downwards
                    point = (left, top + (distance - 2 * width - height))
                if margin <= point[0] <= tactical.w - margin and margin <= point[1] <= tactical.h - margin:
                    points.append((round(point[0]), round(point[1])))
        return points

    def front_point(self, reach=190):
        """A spot beside the base toward the map center (where the enemy comes from), for the
        army to wait without standing on building sites. Needs the radar; else below the base."""
        tactical = self.layout.tactical
        x, y = self.home
        radar = self.world.radar.value
        if radar is not None and radar.online and self.radar_home is not None:
            cx, cy = radar.center
            dx, dy = cx - self.radar_home[0], cy - self.radar_home[1]
            length = (dx * dx + dy * dy) ** 0.5
            if length >= 3:
                x, y = x + reach * dx / length, y + reach * dy / length
        else:
            y += reach * 0.6
        margin = 60
        return (round(min(max(x, margin), tactical.w - margin)), round(min(max(y, margin), tactical.h - margin)))

    def in_view(self, point, margin=60):
        tactical = self.layout.tactical
        return margin <= point[0] <= tactical.w - margin and margin <= point[1] <= tactical.h - margin

    def frame_copy(self):
        return None if self.last_canvas is None else self.last_canvas.copy()

    def find_cursor(self, canvas, near=None, radius=70, region=None):
        if self.cursor_finder is None or canvas is None:
            return None
        if near is None:
            return self.cursor_finder.find(canvas, region)
        x, y = int(near[0]), int(near[1])
        # The hotspot is the sprite's top-left corner: the window must also hold the sprite below
        # and right of it (live game 8: a cursor 30 px low was never found, and every sidebar click
        # after that landed one slot too low).
        sprite = getattr(self.cursor_finder, 'sprite', None)
        sprite_h, sprite_w = sprite.shape[:2] if sprite is not None else (24, 24)
        left, top = max(0, x - radius), max(0, y - radius)
        width = min(canvas.shape[1], x + radius + sprite_w) - left
        height = min(canvas.shape[0], y + radius + sprite_h) - top
        return self.cursor_finder.find(canvas, (left, top, width, height))


def lane_view(snapshot, lane):
    return snapshot['lanes'][lane]


TAB_KEYS = {'buildings': 'q', 'defenses': 'w', 'infantry': 'e', 'vehicles': 'r'}


def select_tab(ctx, lane):
    """E/R switch the unit tabs without moving the pointer; clicking the tab is the fallback.

    Structure and defense tabs are clicked: their keys (Q/W) also pick up a finished
    building for placement, which only place_building wants.
    """
    if ctx.world.active_tab.value == lane:
        return
    if ctx.keyboard_tabs and lane in ('infantry', 'vehicles'):
        yield Act([Step.key(TAB_KEYS[lane])])
        if (yield Poll(lambda s, v: s['tab'] == lane, 0.6, f'{lane} tab key')):
            ctx.keyboard_tab_misses = 0
            return
        # A miss falls back to a click this time only: in live game 2 the misses came from a
        # destroyed factory (its tab was gone), and disabling the keys slowed every later switch.
        ctx.keyboard_tab_misses += 1
    yield Act([Step.click(ctx.tab_point(lane)), Step.move(ctx.park)])
    yield Wait(lambda s, v: s['tab'] == lane, 1.5, f'{lane} tab')


def calibrate_pointer(ctx):
    """Find the game cursor over the sidebar (always an arrow there), then learn the motion gain.

    Two large moves pin the game cursor in the canvas' top right corner (over the sidebar,
    whatever is selected), and a move back down-left leaves the edges in the same transaction.
    Nudges are split into short jumps so OS clipping cannot truncate the measured motion. It ends over the
    cameo grid, where the cursor is the big arrow whatever its height was before (over the
    online radar the game draws a small one: live game 17 lost the cursor there for good).
    Gain is measured once, by moves inside the sidebar; later calibrations (after a pause
    or a lost cursor) only find the cursor again.
    """
    sidebar = ctx.layout.sidebar.tuple()
    yield Act([Step.nudge(1100, -1100), Step.nudge(1100, -1100), Step.nudge(-70, 250)], commands=0)
    yield Poll(lambda s, v: False, 0.08)
    position = yield Look(lambda s, v: ctx.find_cursor(ctx.last_canvas, region=sidebar))
    if position is None:
        raise TaskFailed('Cannot see the game cursor over the sidebar (is the pointer captured?).')
    if ctx.gain_measured:
        ctx.pointer.observe(position)
        yield Act([Step.move(ctx.park)], commands=0)
        return
    measured = []
    for dx, dy in ((0, 260 if position[1] < 400 else -260), (-25, 0)):
        yield Act([Step.nudge(dx, dy)], commands=0)
        yield Poll(lambda s, v: False, 0.12)
        expected = (position[0] + dx, position[1] + dy)
        moved = yield Look(lambda s, v: ctx.find_cursor(ctx.last_canvas, expected, radius=90))
        if moved is None:
            moved = yield Look(lambda s, v: ctx.find_cursor(ctx.last_canvas, region=sidebar))
        if moved is None:
            raise TaskFailed('Lost the game cursor during calibration.')
        gains = [c / (m - p) for c, m, p in zip((dx, dy), moved, position) if c and abs(m - p) >= 8]
        if not gains:
            raise TaskFailed('The game cursor did not follow injected motion (is the pointer captured?).')
        measured.append(sum(gains) / len(gains))
        position = moved
    large, small = measured
    if not 0.3 <= large <= 3.0:
        raise TaskFailed(f'Implausible pointer gain {large:.2f}.')
    if abs(large / small - 1) > 0.2:
        raise TaskFailed(f'Pointer motion is not linear (gains {small:.2f} vs {large:.2f}). '
                         'Disable mouse acceleration in the game options, then resume.')
    ctx.pointer.gain = large
    ctx.gain_measured = True
    ctx.pointer.observe(position)
    ctx.world.note('pointer', gain=round(large, 3), small_gain=round(small, 3), at=list(position))
    yield Act([Step.move(ctx.park)], commands=0)       # off the icons: no hover tooltips
    if ctx.world.mcv_deployed.value:
        yield from center_base(ctx)


def center_base(ctx):
    """Press H (center on base) and remember where the construction yard appears, on screen
    and on the radar (the camera box then surrounds the base)."""
    yield Act([Step.key('h')])
    yield Poll(lambda s, v: False, 0.35)
    ctx.camera = 'home'
    found = yield Look(lambda s, v: find_base(ctx.last_canvas, ctx.home if ctx.home_known else None))
    if found is not None:
        ctx.home, ctx.home_known = found, True
        ctx.world.set('home', list(found))
    radar = ctx.world.radar.value
    if radar is not None and radar.online and radar.view is not None:
        ctx.radar_home = radar_home(radar, ctx.radar_home)


def radar_home(radar, previous=None):
    """Our base on the radar: the largest group of our dots near the camera box (after H),
    or the camera box center."""
    center = radar.view_center
    near = [(area, point) for area, point in radar.own
            if abs(point[0] - center[0]) <= 12 and abs(point[1] - center[1]) <= 12]
    return max(near)[1] if near else (previous or center)


def _deployed(s, v):
    return bool(lane_view(s, 'buildings')['available'])


def _near(point, target, distance):
    return point is not None and abs(point[0] - target[0]) + abs(point[1] - target[1]) <= distance


def _record_deploy(ctx, how, point=None):
    ctx.world.structures['construction_yard'] = 1
    ctx.world.set('mcv_deployed', True)
    if point is not None:
        ctx.home, ctx.home_known = point, True
    ctx.world.note('deployed', by=how, at=list(point) if point else None)


def deploy_mcv(ctx, keys=True):
    """Deploy the MCV as fast as possible, wherever it stands.

    At match start N selects our only unit, S stops it and D deploys it: keyboard only, so it needs no
    pointer calibration and happens on the first tick. D is safe only here: on a
    construction yard it packs the base up ("MCV Repacks"), so the executor allows it only
    while no construction yard exists. If the keys do not deploy it (the MCV moved, another
    unit was selected, its spot is blocked), the MCV is found on screen and deployed by
    clicks, driving to open ground first when its spot is blocked.
    """
    if keys:
        yield Act([Step.key('n'), Step.wait(150), Step.key('s'), Step.wait(150), Step.key('d')])
        if (yield Poll(_deployed, 1.5, 'construction yard')):
            _record_deploy(ctx, 'keys')
            yield from center_base(ctx)
            return
    if ctx.pointer is not None and getattr(ctx.pointer, 'position', 0) is None:
        raise TaskFailed('MCV did not deploy by keys; clicking it needs a calibrated pointer. Retrying soon.')
    yield from deploy_mcv_by_clicks(ctx)


def deploy_mcv_by_clicks(ctx, moves=3):
    """Select the MCV (found by house color) and click it again to deploy; while its spot is
    blocked, drive it to the most open ground nearby (never a spot tried before) and retry.

    Only something that still looks like an MCV is clicked: clicking a selected construction
    yard would pack it up again.
    """
    point = yield Look(lambda s, v: find_mcv(ctx.last_canvas))
    if point is None:
        yield Act([Step.key('h')])               # center the view in case the MCV is off-screen
        yield Poll(lambda s, v: False, 0.3)
        point = yield Look(lambda s, v: find_mcv(ctx.last_canvas))
    if point is None:
        raise TaskFailed('Cannot find the MCV on screen.')
    yield Act([Step.click(point)])               # select
    yield Poll(lambda s, v: False, 0.15)
    tried = []
    for move in range(moves + 1):
        if not (yield Look(lambda s, v: s.get('sidebar_empty'))):
            break                                # a construction yard appeared meanwhile
        here = yield Look(lambda s, v: find_mcv(ctx.last_canvas))
        if here is None:
            break
        yield Act([Step.click(here)])            # deploy (the MCV is selected)
        if (yield Poll(_deployed, 1.2, 'construction yard')):
            _record_deploy(ctx, 'clicks', here)
            yield from center_base(ctx)
            return
        if move == moves:
            break
        tried.append(here)
        spots = yield Look(lambda s, v: open_spots(ctx.last_canvas, here, avoid=tried))
        if not spots:
            raise TaskFailed('The MCV cannot deploy here and no open ground is in view.')
        tried.append(spots[0])
        yield Act([Step.click(spots[0])])        # move order: the MCV is still selected
        arrived = yield Poll(lambda s, v: _near(find_mcv(ctx.last_canvas), spots[0], 30), 6.0, 'MCV arrival')
        ctx.world.note('mcv-moved', to=list(spots[0]), arrived=bool(arrived))
    if (yield Look(_deployed)):
        _record_deploy(ctx, 'clicks', point)
        yield from center_base(ctx)
        return
    raise TaskFailed('MCV did not deploy (blocked, or not selected). Retrying soon.')


def queue_item(ctx, item, count=1):
    """Queue `count` of an item (a click on its cameo each) and confirm that production started."""
    lane = ctx.lane(item)
    yield from select_tab(ctx, lane)
    index = yield Look(lambda s, v: ctx.slot_index(item, ('available',)))
    if index is None:
        raise TaskFailed(f'{item} is not available to build now.')
    cost = ctx.world.item(item)['cost']
    units = lane in ('infantry', 'vehicles')
    spendable = ctx.world.spendable()
    if units and spendable is not None and spendable < cost * count:
        raise TaskFailed(f'Credits changed before queueing {item}; need {cost * count}, have {spendable}.')
    entry = ctx.world.ledger.add('queue', lane, item, cost * count)
    clicks = [Step.click(ctx.slot_point(index))]
    for _ in range(count - 1):
        clicks += [Step.wait(60), Step.click(ctx.slot_point(index))]
    yield Act(clicks + [Step.move(ctx.park)], commands=count)

    def started(s):
        view = lane_view(s, lane)
        # Structures build one at a time; a unit click may instead join the lane's queue,
        # which shows as a count on its cameo while another unit is in production.
        if view['status'] in ('building', 'ready') and (view['item'] == item or units):
            return True
        if units and any(slot.item == item and slot.count for slot in ctx.world.slots[lane].value or []):
            return True
        return None
    try:
        yield Expect(started, 2.5, 'production started')
    except GeneratorExit:
        ctx.world.ledger.close(entry, 'cancelled')
        raise
    ctx.world.ledger.close(entry, 'confirmed')
    if units:
        ctx.world.units_built[item] = ctx.world.units_built.get(item, 0) + count


def cancel_ready(ctx, item):
    """Cancel a finished building that could not be placed: a right click on its Ready cameo
    refunds it and frees the construction lane."""
    lane = ctx.lane(item)
    yield from select_tab(ctx, lane)
    index = yield Look(lambda s, v: ctx.slot_index(item, ('ready',)))
    if index is None:
        return                                   # already placed or gone
    yield Act([Step.click(ctx.slot_point(index), 'right'), Step.move(ctx.park)])
    yield Wait(lambda s, v: not (lane_view(s, lane)['item'] == item and lane_view(s, lane)['status'] == 'ready'),
               2.0, f'{item} cancelled')
    ctx.world.note('cancelled', item=item, reason='no valid spot after repeated placement attempts')


def shrouded(canvas, point, half=24):
    """Is the ground around a point still black shroud? No footprint is drawn there."""
    if canvas is None:
        return False
    x, y = int(point[0]), int(point[1])
    patch = canvas[max(0, y - half):y + half, max(0, x - half):x + half]
    return patch.size == 0 or float(patch.mean()) < 25


def _step_away(point, ghost):
    """One cell from `point` away from the footprint's blocked (red) part."""
    dx = ghost.green_center[0] - ghost.red_center[0]
    dy = ghost.green_center[1] - ghost.red_center[1]
    step_x = CELL[0] * (1 if dx > 0 else -1) if abs(dx) >= 10 else 0
    step_y = CELL[1] * (1 if dy > 0 else -1) if abs(dy) >= 5 else 0
    return None if (step_x, step_y) == (0, 0) else (point[0] + step_x, point[1] + step_y)


def factory_access_open(canvas, point):
    """A green factory footprint still needs free ground beyond its vehicle door."""
    if canvas is None:
        return False
    ground = ground_mask(canvas)
    return all(openness(ground, (point[0] + 60, point[1] + dy), half=(50, 25)) >= .8
               for dy in (45, 75))


def place_building(ctx, item, ring=0, max_probes=30, front=None, site=None):
    """Pick up the finished building, move its footprint around the base, click only where it is all green.

    Q/W switch to the tab and pick up a completed building; if no footprint follows the
    cursor, the Ready cameo is clicked instead. A frame taken with the cursor over the
    radar (no footprint on the map) is the reference the footprint is measured against.
    Blocked footprints move one cell away from their red part before the next spot is tried.
    A fresh home-view raid point can prioritize defensive sites on its side of the base.
    """
    lane = ctx.lane(item)
    if site is None:
        yield from center_base(ctx)              # known camera: the base is where H puts it
        extent = yield Look(lambda s, v: base_extent(ctx.last_canvas, ctx.home if ctx.home_known else None))
    else:
        extent = site
    yield Act([Step.key(TAB_KEYS[lane]), Step.move(ctx.park)])
    yield Wait(lambda s, v: s['tab'] == lane, 1.5, f'{lane} tab')
    yield Poll(lambda s, v: False, 0.15)
    reference = yield Look(lambda s, v: ctx.frame_copy())

    def placed(s, v):
        view = lane_view(s, lane)
        return s['tab'] == lane and not (view['item'] == item and view['status'] == 'ready')

    points = ctx.placement_candidates(extent, ring)
    rear_priority = None
    if lane == 'defenses' and site is None:
        front = front if front is not None else ctx.front_point()
        points.sort(key=lambda point: (point[0] - front[0]) ** 2 + (point[1] - front[1]) ** 2)
    elif (site is None and ctx.world.item(item).get('placement') == 'rear'
          and ctx.world.radar.value is not None and ctx.world.radar.value.online
          and ctx.radar_home is not None):
        # Keep vulnerable technology behind the base, away from the known approach.
        hx, hy = ctx.home
        fx, fy = ctx.front_point()
        rear_priority = lambda point: (
            (point[0] - hx) * (fx - hx) + (point[1] - hy) * (fy - hy) > 0,
            (point[0] - hx) ** 2 + (point[1] - hy) ** 2)
        points.sort(key=rear_priority)
    candidates = deque(points)
    tried, entered, probes = set(), 0, 0
    while candidates and probes < max_probes:
        point = candidates.popleft()
        cell = (point[0] // 20, point[1] // 20)
        # Shroud draws no footprint (live game 2026-09-25: probes there read as "not holding it").
        if cell in tried or not ctx.in_view(point) or shrouded(ctx.last_canvas, point):
            continue
        tried.add(cell)
        if site is None:
            if item == 'war_factory' and not factory_access_open(reference, point):
                continue
            if (item != 'war_factory' and ctx.factory_exit is not None and
                    abs(point[0] - ctx.factory_exit[0]) + 2 * abs(point[1] - ctx.factory_exit[1]) < 180):
                continue
        yield Act([Step.move(point)], commands=0)
        yield Poll(lambda s, v: False, 0.1)
        if not (yield Look(lambda s, v: s['tab'] == lane)):
            raise TaskFailed(f'{lane} tab is no longer active; stopping placement of {item}.')
        ghost = yield Look(lambda s, v: read_ghost(ctx.last_canvas, reference, point))
        probes += 1
        if not ghost.seen:
            # Not holding the building (the key did not pick it up): click its Ready cameo.
            if entered >= 2:
                raise TaskFailed(f'Placement mode for {item} did not start (no footprint under the cursor).')
            index = yield Look(lambda s, v: ctx.slot_index(item, ('ready',)))
            if index is None:
                raise TaskFailed(f'{item} is not ready to place.')
            yield Act([Step.click(ctx.slot_point(index)), Step.move(ctx.park)])
            yield Poll(lambda s, v: False, 0.15)
            reference = yield Look(lambda s, v: ctx.frame_copy())
            entered += 1
            tried.discard(cell)
            candidates.appendleft(point)
            continue
        if ghost.valid:
            yield Act([Step.click(point), Step.move(ctx.park)])
            if (yield Poll(placed, 0.6, 'placement')):
                ctx.world.structures[item] = ctx.world.structures.get(item, 0) + 1
                if item == 'war_factory' and site is None:
                    ctx.factory_exit = (point[0] + 60, point[1] + 60)
                ctx.world.note('placed', item=item, point=list(point), probes=probes)
                return
            reference = yield Look(lambda s, v: ctx.frame_copy())   # still holding it: something moved in
            continue
        if ghost.red_center and ghost.green_center:
            nearby = _step_away(point, ghost)
            if nearby is not None:
                candidates.appendleft(nearby)
                if rear_priority is not None:
                    # A blocked-footprint detour must not bypass untried rear sites.
                    candidates = deque(sorted(candidates, key=rear_priority))
    # Leave placement mode without building: right-click cancels placement.
    yield Act([Step.click(ctx.home, 'right'), Step.move(ctx.park)])
    raise TaskFailed(f'No valid spot found for {item} around the base ({probes} spots checked).')


def place_near_ally(ctx, item, ally, max_probes=8):
    """Try a revealed teammate's construction radius, then place at home if it is blocked."""
    from .army import radar_jump
    from .perception import colors
    from .perception.units import house_blobs
    ctx.camera = 'away'
    try:
        moved = yield from radar_jump(ctx, ally)
        if moved:
            blobs = yield Look(lambda s, v: house_blobs(ctx.last_canvas, ctx.layout.tactical.tuple(),
                    mask=colors.ally_mask(ctx.layout.tactical.crop(ctx.last_canvas),
                                          compressed=ctx.compressed_frames).astype('uint8')))
            buildings = [b for b in blobs if b[0] >= 400]
            if buildings:
                _, _, (x, y, w, h) = buildings[0]
                try:
                    yield from place_building(ctx, item, max_probes=max_probes, site=(x, y, x + w, y + h))
                except TaskFailed:
                    ctx.world.note('ally-placement', item=item, result='blocked', target=list(ally))
                else:
                    ctx.world.note('ally-placement', item=item, result='placed', target=list(ally))
                    yield from center_base(ctx)
                    return
        yield from place_building(ctx, item)
    finally:
        # An interrupted remote task cannot leave home-camera threat detection active.
        if ctx.camera != 'home':
            ctx.camera = 'lost'
