"""The Chrono Divide player loop: observe -> remember -> decide -> act -> verify, ~20 Hz.

Modes:
  watch      perception and world model only; never sends input
  recommend  also asks System One and computes the next action, shows it, never sends input
  execute    acts, but only while the match is actually being played (HUD visible,
             no game menu, pointer held by the game, window in front)

Which game: a client profile (client.py; Chrono Divide unless the setup chose another) says
how to find its window and view, its house colors, and how the mouse reaches it (browser
pointer lock, or ordinary absolute input). The layout follows the view's size.
"""

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time

import cv2

from desktop import input as di
from desktop.input import Gate, clip_rect, cursor
from desktop.target import DesktopPause, enable_dpi
from . import playbook
from .army import Army
from .broker import Broker, server_routes
from .calibration import Setup, load_setup, pick_window, process_name, save_setup
from .client import ClientProfile
from .executor import AbsolutePointer, Executor, Guard, LockedPointer, NotPlaying, Trust
from .intel import Intel
from .intent import StandingIntent
from .perception import view as views
from .perception.detector import Detector
from .perception.modes import playing, pointer_locked
from .perception.radar import annotate as annotate_radar
from .perception.reader import Perception
from .perception.sidebar import annotate
from .perception.scene import SceneWorker, draw_scene, summary
from .recorder import GameView
from .reflex import Reflex
from .repair import Repair
from .skills import Context
from .tasks import TaskRunner
from .window import pointer_state
from .world import World

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / '.cache' / 'chronodivide' / 'runs'
UNIT_SCAN = 0.2                  # seconds between scene scans for the inspector (30-50 ms each, own thread)
UNIT_SNAPSHOT = 5.0              # seconds between saved snapshots of recognized units (runs/<time>/units)


ATTACHED_BASE = dict(construction_yard=1, tesla_reactor=3, ore_refinery=2, barracks=1, war_factory=1, radar_tower=1)


def attach_to_match(world, army):
    """Joining a match already under way (a restarted controller): the world learns units and
    buildings only as it makes them, so the army (in Rhinos) and a standard base are assumed."""
    if not army:
        return
    world.units_built['rhino_tank'] = world.units_built.get('rhino_tank', 0) + int(army)
    for item, count in ATTACHED_BASE.items():
        world.structures[item] = max(world.structures.get(item, 0), count)
    world.set('mcv_deployed', True)
    world.note('attached', army=int(army), base=ATTACHED_BASE)


class JsonLog:
    def __init__(self, path):
        self.file = open(path, 'a', encoding='utf-8')
        self.lock = threading.Lock()

    def __call__(self, entry):
        with self.lock:
            self.file.write(json.dumps(entry, default=str) + '\n')
            self.file.flush()

    def close(self):
        self.file.close()


class Player:
    def __init__(self, mode='watch', server='http://127.0.0.1:8000', policy='systemone', tick_hz=20.0,
                 record_fps=0.0, require_lock=None, output=None, autoplay=False, profile=None, target=None,
                 setup=None, strategy=None):
        if mode not in ('watch', 'recommend', 'execute'):
            raise ValueError('mode must be watch, recommend or execute')
        enable_dpi()
        self.mode, self.policy, self.tick_period = mode, policy, 1.0 / tick_hz
        self.setup = setup if setup is not None else load_setup()
        self.profile = profile or ClientProfile.load(self.setup.profile if self.setup else 'chronodivide')
        self.profile.use_colors()
        # Browser pointer lock needs relative motion and the lock as evidence of play; a game
        # that reads the ordinary cursor takes absolute input.
        self.require_lock = self.profile.input == 'locked' if require_lock is None else require_lock
        self.output = Path(output or RUNS / datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = JsonLog(self.output / 'events.jsonl')
        self.decisions = JsonLog(self.output / 'decisions.jsonl')
        self.actions = JsonLog(self.output / 'actions.jsonl')
        self.target = target or pick_window(self.profile, self.setup)
        marked = self._marked_view()
        size = (marked[0].width, marked[0].height) if marked else self.profile.size
        self.view = GameView(self.target, relocate_seconds=None, profile=self.profile, size=size,
                             marked=marked, on_found=self._found_view)
        self.perception = Perception(profile=self.profile, layout=self.profile.layout(*size))
        self.layout = self.perception.layout
        views.use(self.layout)
        self.events(dict(time=time.time(), client=self.profile.name, window=self.target.title, view=list(size),
                         input='locked' if self.require_lock else 'absolute'))
        self.book = playbook.load(strategy) if strategy else playbook.load()
        self.world = World(knowledge=playbook.knowledge(self.book), log=self.events)
        attach_to_match(self.world, os.environ.get('CHRONODIVIDE_ATTACH_ARMY'))
        self.intent = StandingIntent()
        self.gate = Gate()
        self.generation = self.gate.generation
        self.broker = None
        self.server = server if policy == 'systemone' and mode in ('recommend', 'execute') else None
        self.broker_attempt = -1e9
        self.trust = Trust(ROOT / '.cache' / 'chronodivide' / 'skill-trust.json')
        self.guard = Guard(self.target, self.view, self.gate, self.gate.generation,
                           takeover=self._takeover, playing=self._playing_now,
                           deploy_allowed=lambda: bool(self.world.sidebar_empty.value)
                           and not self.world.mcv_deployed.value)
        self.pointer = None
        self.executor = self.runner = None
        if mode == 'execute':
            self.pointer = (LockedPointer(self.guard, cursor=cursor, clip=clip_rect,
                                          relative=self.profile.raw.get('relative_motion', False)) if self.require_lock
                            else AbsolutePointer(self.guard))
            self.executor = Executor(self.guard, self.pointer, log=self.actions)
            self.runner = TaskRunner(self.executor, self.trust, log=self.events)
        self.ctx = Context(self.layout, self.world, self.perception.cursor, self.pointer,
                           build_off_ally=self.profile.raw.get('team_play', {}).get('build_off_ally', False))
        self.intel = Intel(time.monotonic, book=self.book)
        self.army = Army(self.ctx, self.runner, self.intent, log=self.events, intel=self.intel, book=self.book)
        if os.environ.get('CHRONODIVIDE_ATTACH_ARMY'):
            self.army.census_enabled = False      # a joined match's army may be in the field, not home
        self.reflex = Reflex(self.ctx, self.runner, self.broker, self.intent, log=self.events,
                             generation=lambda: self.gate.generation, army=self.army, book=self.book)
        self.repair = Repair(self.ctx, self.runner, self.book, time.monotonic)
        if self.runner:
            self.runner.on_failure = self.reflex.failed
            self.runner.on_done = self.reflex.done
        self.screens = None                # menu screen templates (score screen = the match ended)
        self.pilot = None
        if autoplay and mode == 'execute':
            from .autoplay import MenuPilot
            self.pilot = MenuPilot()       # clicks Continue / Start Game between matches
        self.last_radar_save = 0.0
        self.items = self.world.knowledge['items']                          # names for the unit labels
        # The inspector's scene: units and buildings tracked on their own thread (perception/scene.py);
        # the loop only hands frames over. The unit detector, once trained, runs in the SystemX
        # server when one is up (kept in its memory), else here on the CPU.
        self.last_units_save = 0.0
        self.detector = Detector(server=server) if Detector.available() else None
        self.scene = SceneWorker(built=self._unit_names, items=self.items, detector=self.detector,
                                 period=UNIT_SCAN, on_scan=self._scanned)
        if mode == 'execute':
            from .tools.detector import CENSUS, PairRecorder
            self.ctx.pairs = PairRecorder(CENSUS / self.output.name)     # the detector's training views
        self.match = dict(started=None, ended=None, screen=None)
        self.watch = None
        self._physical = False
        self.stop_event = threading.Event()
        self.status = dict(state='starting', reason='')
        self.preview = None
        self.hud = None
        self.last_mode = None
        self.commands = deque()
        self.timings = deque(maxlen=200)
        self.record_period = 1.0 / record_fps if record_fps else None
        self.last_record = 0.0
        self.frames_saved = 0
        self.last_playing = (False, 'not started')
        self.logged_playing = None
        self.hud_seen = 0.0
        self.pointer_free = False          # HUD on screen and game window in front, pointer not captured
        self.menu_since = None             # when the game's pause menu appeared
        self.menu_by_user = False          # physical input just before it: the user opened it
        self.pending_arm = False

    def _marked_view(self):
        """The view marked in the setup, for the window it was marked in (while its size holds)."""
        setup = self.setup
        if setup is None or not setup.window or setup.window.get('hwnd') != self.target.hwnd:
            return None
        try:
            return setup.marked(self.target.geometry())
        except DesktopPause:
            return None

    def _found_view(self, rect, geometry):
        """The HUD anchors found the view: keep it in the setup, so menus (where they are missing)
        and later runs find it too while the window keeps its size."""
        setup = self.setup or Setup(profile=self.profile.slug)
        if setup.view == [rect.x, rect.y, rect.width, rect.height] and setup.client == [geometry.width, geometry.height] \
                and (setup.window or {}).get('hwnd') == self.target.hwnd:
            return
        setup.remember(self.target, process_name(self.target.pid), rect, geometry)
        try:
            save_setup(setup)
        except OSError:
            return
        self.setup = setup
        self.view.marked = setup.marked(geometry)
        self.events(dict(time=time.time(), setup='view saved', view=setup.view, client=setup.client))

    def _adopt_layout(self, width, height):
        """The game view has another size: take the layout for it. Running tasks are cancelled
        (their points belong to the old view); the base is found again with H."""
        if self.runner is not None:
            self.runner.cancel_all('the game view changed size')
        layout = self.profile.layout(width, height)
        self.perception.use_layout(layout)
        self.ctx.use_layout(layout)
        self.layout = layout
        views.use(layout)
        self.events(dict(time=time.time(), view=[width, height], layout=layout.name))

    # -- control ------------------------------------------------------------
    def _takeover(self):
        return self._physical

    def arm(self):
        """F8: start or resume acting (execute mode only)."""
        if self.mode != 'execute':
            return
        self._physical = False
        self.pause_reason = None
        if self.watch is None:
            from desktop.hooks import watch_physical_input
            self.watch = watch_physical_input(self._on_physical, ignore_keys=(0x77, 0x78, 0x79))
        self.watch.reset_anchor()
        self.gate.start()
        self.guard.generation = self.gate.generation
        self.pointer.resync()
        self.events(dict(time=time.time(), control='armed', generation=self.gate.generation))

    def arm_on_start(self):
        """Arm as soon as the match is actually being played (e.g. pressed while in the menu).

        The physical-input watch starts only then, so clicking Start Game does not count
        as a takeover; from match start on, the user keeps hands off.
        """
        if self.mode != 'execute':
            return
        if self.last_playing[0]:
            self.arm()
            return
        self._physical = False           # resuming ends the takeover pause, even before play resumes
        self.pause_reason = None
        self.pending_arm = True
        self.events(dict(time=time.time(), control='armed-on-start'))

    def pause(self, reason='paused by user'):
        self.pending_arm = False
        self.pause_reason = reason
        self.gate.pause()
        self.status.update(state='paused', reason=f'{reason}. Agent stopped; the game may still be running. '
                                                 'F8 resumes the agent; Esc in game opens its menu.')
        if self.runner:
            self.runner.cancel_all(reason)
        if self.pointer:
            self.pointer.resync()
        self.events(dict(time=time.time(), control='paused', reason=reason))

    def _on_physical(self, kind):
        if self.gate.running:
            self._physical = True
            self.gate.pause()
            self.events(dict(time=time.time(), control='paused', reason=f'physical input ({kind})'))

    def taken_over(self):
        """Paused because someone used the physical mouse or keyboard."""
        return self._physical

    def in_front(self):
        return di.user32.GetForegroundWindow() == self.target.hwnd

    def paused_internally(self):
        """Monotonic time of a pause the agent caused itself (a guard or input problem), while it
        lasts; None otherwise."""
        since = getattr(self, 'internal_pause_at', None)
        if since is None or self.gate.running or self._physical or self.pending_arm:
            return None
        return since

    def last_physical(self):
        """Monotonic time of the latest physical input seen while watching, or None."""
        seen = self.watch.last_physical if self.watch else None
        return seen[1] if seen else None

    def wants_pointer(self):
        """Armed (or waiting to arm) while the HUD is up but the game does not hold the pointer."""
        return self.mode == 'execute' and (self.gate.running or self.pending_arm) and self.pointer_free

    def menu_open(self):
        return self.mode == 'execute' and self.menu_since is not None

    def recapture_pointer(self):
        """Let the game capture the pointer again: it requests pointer lock only from a click
        on its canvas (the map area; sidebar and menus are separate page elements). A right
        click first clears any selection, so the left click cannot become an order.
        For unattended runs only (see Supervisor)."""
        point = self.layout.points.get('recapture', (425, 650))
        x, y = self.guard.desktop_point(point)
        self.target.check_point(x, y, foreground=True)
        desktop = di.virtual_desktop()
        sent = di.send(di.mouse_events((x, y), desktop, 'right'))
        time.sleep(0.05)
        sent += di.send(di.mouse_events((x, y), desktop, 'left', move=False))
        self.pointer.resync()
        self.events(dict(time=time.time(), control='recapture-pointer', point=list(point), sent=sent))

    def resume_menu(self):
        """Click Resume Mission in the game's pause menu (menus take absolute clicks)."""
        point = self.layout.points.get('resume_mission', (940, 713))
        x, y = self.guard.desktop_point(point)
        self.target.check_point(x, y, foreground=True)
        sent = di.send(di.mouse_events((x, y), di.virtual_desktop(), 'left'))
        self.pointer.resync()
        self.events(dict(time=time.time(), control='resume-menu', point=list(point), sent=sent))

    def menu_click(self, point, canvas_rect, geometry, rest=0.3):
        """Hover, then click a menu button (menus take absolute clicks; the pointer is free there)."""
        from .window import bring_to_front
        if di.user32.GetForegroundWindow() != self.target.hwnd:
            bring_to_front(self.target)
        x, y = geometry.x + canvas_rect.x + point[0], geometry.y + canvas_rect.y + point[1]
        self.target.check_point(x, y, foreground=True)
        desktop = di.virtual_desktop()
        di.send(di.mouse_events((x, y), desktop))
        time.sleep(rest)
        di.send(di.mouse_events((x, y), desktop, 'left', move=False))
        if self.pointer is not None:
            self.pointer.resync()

    def stop(self):
        self.stop_event.set()

    # -- loop ---------------------------------------------------------------
    def _playing_now(self):
        return self.last_playing

    def _ensure_broker(self):
        """Connect to System One when it is ready; until then the reflex uses its rule defaults."""
        if self.server is None or self.broker is not None:
            return
        now = time.monotonic()
        if now - self.broker_attempt < 3.0:
            return
        self.broker_attempt = now
        try:
            from systemone_client import SystemOneClient
            with SystemOneClient(self.server, timeout=2) as client:
                if not client.health().get('ready'):
                    return
            self.broker = Broker(server_routes(self.server), memo_seconds=self.book.systemone['memo_seconds'],
                                 log=self.decisions,
                                 generation=lambda: self.gate.generation if self.mode == 'execute' else 0)
            self.reflex.broker = self.broker
            self.events(dict(time=time.time(), systemone='connected', routes=len(self.broker.endpoints)))
        except Exception as error:
            self.events(dict(time=time.time(), systemone='unavailable', error=str(error)[:120]))

    def run(self):
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    self.tick()
                except DesktopPause as error:
                    self.last_playing = (False, str(error))
                    self.pointer_free = False
                    self.menu_since = None
                    self.status.update(state='waiting', reason=str(error))
                    time.sleep(0.25)
                elapsed = time.monotonic() - started
                time.sleep(max(0.0, self.tick_period - elapsed))
        finally:
            self.close()

    def tick(self):
        self._ensure_broker()
        t0 = time.perf_counter()
        frame = self.view.read()
        if (self.view.canvas.width, self.view.canvas.height) != self.layout.canvas:
            self._adopt_layout(self.view.canvas.width, self.view.canvas.height)
        canvas = frame.pixels
        canvas_rect, geometry = self.view.canvas, frame.geometry
        self.ctx.last_canvas = canvas
        t1 = time.perf_counter()
        mode, readings, _ = self.perception.read(canvas, self.world)
        self.hud = dict(mode, required=self.perception.modes.required)
        if mode['mode'] == 'menu':
            if self.menu_since is None:
                self.menu_since = time.monotonic()
                last = self.last_physical()
                self.menu_by_user = last is not None and self.menu_since - last < 3.0
        else:
            self.menu_since = None
        if mode['mode'] == 'hud':
            self.hud_seen = time.monotonic()
        elif mode['mode'] != 'menu' and time.monotonic() - max(self.hud_seen, self.view.located_at) > 1.5:
            # The page can move the canvas without moving the window (a browser info bar
            # appearing or closing): look for the canvas again instead of trusting its old place.
            self.view.canvas = None
        pointer = pointer_state()
        foreground = pointer['foreground'] == self.target.hwnd
        self.pointer_free = (self.require_lock and foreground and mode['mode'] == 'hud'
                             and not pointer_locked(pointer, frame.geometry))
        if self.require_lock:
            self.last_playing = playing(mode, pointer, frame.geometry, foreground)
        else:
            self.last_playing = ((mode['mode'] == 'hud' and foreground),
                                 'playing' if mode['mode'] == 'hud' else mode['mode'])
        self._track_match(canvas, mode)
        if self.last_playing != self.logged_playing:
            self.events(dict(time=time.time(), playing=self.last_playing[0], reason=self.last_playing[1],
                             pointer=pointer, foreground=foreground))
            self.logged_playing = self.last_playing
        if self.pilot is not None and mode['mode'] not in ('hud', 'menu'):
            action = self.pilot.step(canvas, self.last_physical(), time.monotonic())
            if action is not None:
                self.menu_click(action[1], canvas_rect, geometry)
                self.events(dict(time=time.time(), autoplay='click', screen=action[0], point=list(action[1])))
        if mode['mode'] == 'hud':
            self.scene.submit(canvas, self.layout.tactical.tuple())
        else:
            self.scene.clear()
        radar = self.world.radar.value
        if radar is not None and radar.online and self.ctx.radar_home is None and radar.own:
            self.ctx.radar_home = radar.own[0][1]     # until H confirms it: our largest group of dots
        t2 = time.perf_counter()
        snapshot = self.world.snapshot()
        ok, reason = self.last_playing
        if self.pending_arm and ok:
            self.pending_arm = False
            self.arm()
        acting = self.mode == 'execute' and self.gate.running and ok
        if self._physical and self.mode == 'execute':
            self.status.update(state='paused', reason='You used the mouse/keyboard. Agent stopped; the game may '
                                                     'still be running. F8 resumes; Esc in game opens its menu.')
        plan = None
        if ok or self.mode == 'recommend':
            if self.pointer is not None and acting:
                self._resync_cursor(canvas)
            if not self.repair.step(snapshot, act=acting):
                plan = self.reflex.step(snapshot, act=acting)
                self.army.step(snapshot, act=acting)
            if acting:
                try:
                    self.runner.tick(snapshot, self.view)
                except NotPlaying as error:
                    self.status.update(state='waiting', reason=str(error))
                except DesktopPause as error:
                    self.pause(str(error))
                    self.internal_pause_at = time.monotonic()
                    self._save_failure(canvas, 'pause')
        t3 = time.perf_counter()
        self._count_commands()
        self.timings.append(dict(capture=(t1 - t0) * 1000, perceive=(t2 - t1) * 1000, act=(t3 - t2) * 1000))
        if self.mode != 'execute' or not self._physical:
            if not ok and self.pending_arm:
                self.status.update(state='ready', reason='Armed: acts the moment the match starts. '
                                                         'Hands off after starting it.')
            elif not ok:
                self.status.update(state='waiting', reason=f'Not acting: {reason}.')
            elif self.mode == 'execute' and not self.gate.running:
                paused = getattr(self, 'pause_reason', None)
                self.status.update(state='paused' if paused else 'ready',
                                   reason=f'{paused or "Agent is not armed"}. The match is still running. '
                                          'F8 resumes the agent; Esc in game opens its menu.')
            else:
                self.status.update(state='acting' if acting else self.mode, reason='')
        self._update_preview(canvas, readings, mode, plan)
        self._record(canvas)
        if mode['mode'] != self.last_mode:
            self.events(dict(time=time.time(), mode=mode['mode'], playing=ok, reason=reason))
            self.last_mode = mode['mode']

    def _track_match(self, canvas, mode):
        """Log when the match starts (HUD) and ends (score screen), and save radar crops
        every 10 s while the radar is online (they calibrate radar reading offline)."""
        now = time.time()
        if mode['mode'] == 'hud':
            if self.match['started'] is None or self.match['ended'] is not None:
                self.match = dict(started=now, ended=None, screen=None)
                self.events(dict(time=now, match='started'))
            radar = self.world.radar.value
            if radar is not None and radar.online and now - self.last_radar_save >= 10.0:
                self.last_radar_save = now
                folder = self.output / 'radar'
                folder.mkdir(exist_ok=True)
                cv2.imwrite(str(folder / f'radar-{int(now)}.png'), self.layout.regions['radar'].crop(canvas))
            return
        if self.match['started'] is None or self.match['ended'] is not None or mode['mode'] == 'menu':
            return
        if self.screens is None:
            from .tools.menu import load
            try:
                self.screens = load()
            except Exception:
                self.screens = {}
        from .tools.menu import identify
        screen = identify(canvas, self.screens) if self.screens else None
        if screen == 'score':
            self.match.update(ended=now, screen=screen)
            army = self.army.status()
            self.events(dict(time=now, match='ended', minutes=round((now - self.match['started']) / 60, 1),
                             waves=army['waves'], attacking=army['attacking'], failed_attacks=army['failures'],
                             units=dict(self.world.units_built), structures=dict(self.world.structures)))
            cv2.imwrite(str(self.output / f'score-{int(now)}.png'), canvas)

    def _unit_names(self):
        """Unit types we built (with counts): the candidates for our selected units' names. While the
        scouts walk, only they are selected (called on the scene thread)."""
        built = self.army.combat_built()
        task = self.runner.tasks.get('army') if self.runner is not None else None
        if self.army.scouts_ready and task is not None and task.name in ('army:explore', 'army:scout'):
            scouts = self.book['units']['scouts']
            built = {scouts['item']: scouts['count']}
        return built

    def _scanned(self, canvas, tracks):
        """After each scene scan (scene thread): now and then a snapshot of the named units, for
        reviewing the run (runs/<time>/units)."""
        now = time.monotonic()
        if now - self.last_units_save < UNIT_SNAPSHOT or not any(t.side == 'ours' and t.name for t in tracks):
            return
        self.last_units_save = now
        folder = self.output / 'units'
        folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / f'units-{int(time.time() * 1000)}.jpg'),
                    draw_scene(canvas.copy(), tracks, self.items, now), [cv2.IMWRITE_JPEG_QUALITY, 85])

    @property
    def units(self):
        """The inspector's tracks (perception/scene.py)."""
        return self.scene.snapshot()[0]

    def units_view(self):
        """What the inspector shows about the scene: our units by label, the enemy's by kind."""
        tracks, stats = self.scene.snapshot()
        view = summary(tracks, self.items)
        view.update(ms=stats['ms'], scans=stats['scans'], shift=stats['shift'], error=self.scene.error,
                    detector=self.detector.status() if self.detector else None,
                    pairs=self.ctx.pairs.saved if self.ctx.pairs is not None else None)
        return view

    def _resync_cursor(self, canvas):
        position = getattr(self.pointer, 'position', None)
        if position is None:
            return
        if self.executor is not None and time.monotonic() - self.executor.last_input < 0.12:
            return      # the frame may predate the last move (live game 23: 262 corrections to stale spots)
        found = self.ctx.find_cursor(canvas, position, radius=40)
        if found is None:
            # Not near where it should be: after a second of misses, look over the whole view once.
            # Over the sidebar the cursor is always an arrow (a small one over the online radar), so
            # a miss there means it is lost (live games 15 and 16: 110-1050 px off, and every click
            # until the next check missed): look at once.
            self.cursor_misses = getattr(self, 'cursor_misses', 0) + 1
            sidebar = position[0] >= self.layout.sidebar.x
            if self.cursor_misses % 20 == 0 or (sidebar and self.cursor_misses >= 2):
                found = self.ctx.find_cursor(canvas)
                if found is None and sidebar and self.cursor_misses >= 6:
                    self.pointer.resync()             # unknown: the next task recalibrates first
                    self.cursor_misses = 0
                    self.events(dict(time=time.time(), cursor_lost=list(position)))
                    return
        if found is not None:
            self.cursor_misses = 0
            drift = abs(found[0] - position[0]) + abs(found[1] - position[1])
            self.pointer.observe(found)
            if drift > 6:
                self.events(dict(time=time.time(), cursor_drift=round(drift, 1), expected=position, found=found))

    def _count_commands(self):
        now = time.monotonic()
        total = self.executor.counts['commands'] if self.executor else 0
        self.commands.append((now, total))
        while self.commands and now - self.commands[0][0] > 60:
            self.commands.popleft()

    def apm(self):
        if len(self.commands) < 2:
            return 0.0
        (t0, c0), (t1, c1) = self.commands[0], self.commands[-1]
        return (c1 - c0) * 60.0 / max(1.0, t1 - t0)

    def _update_preview(self, canvas, readings, mode, plan):
        image = canvas.copy()
        if readings:
            annotate(image, self.layout, readings)
        for name, box in self.layout.regions.items():
            if name in ('credits', 'power', 'radar'):
                cv2.rectangle(image, (box.x, box.y), (box.x + box.w - 1, box.y + box.h - 1), (255, 180, 0), 1)
        position = getattr(self.pointer, 'position', None)
        if position is not None:
            x, y = map(int, position)
            cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.drawMarker(image, self.ctx.home, (255, 0, 255), cv2.MARKER_DIAMOND, 14, 1)
        annotate_radar(image, self.world.radar.value, self.army.campaign.target, self.ctx.radar_home)
        if self.intel.enemy_base is not None:
            cv2.drawMarker(image, tuple(map(int, self.intel.enemy_base)), (0, 165, 255), cv2.MARKER_STAR, 12, 1)
        if self.army.seen is not None:
            cv2.circle(image, self.army.seen[0], 24, (255, 128, 0), 2)
        self.preview = image
        self.plan = plan[0] if plan else None

    def _record(self, canvas):
        if self.record_period is None:
            return
        now = time.monotonic()
        if now - self.last_record >= self.record_period:
            self.last_record = now
            frames = self.output / 'frames'
            frames.mkdir(exist_ok=True)
            cv2.imwrite(str(frames / f'frame-{self.frames_saved:06d}.png'), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            self.frames_saved += 1

    def _save_failure(self, canvas, kind):
        folder = self.output / 'failures'
        folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / f'{kind}-{int(time.time() * 1000)}.png'), canvas)

    def view_state(self):
        """Plain data for the controller UI."""
        timings = list(self.timings)
        avg = {k: round(sum(t[k] for t in timings) / len(timings), 1) for k in ('capture', 'perceive', 'act')} \
            if timings else {}
        now = time.monotonic()
        facts = self.reflex.strategy_facts(now)[0]                   # what System One is told, in its words
        return dict(status=dict(self.status), mode=self.mode, policy=self.policy, playing=self.last_playing,
                    hud=self.hud,
                    world=self.world.snapshot(), intent=self.intent.view(), plan=getattr(self, 'plan', None),
                    tasks=self.runner.view() if self.runner else None,
                    systemone=self.broker.summary() if self.broker else None,
                    army=dict(self.army.status(now), attack=facts['attack'], threat=facts['base_threat'],
                              attack_size=self.reflex.attack_size(now), seen=self.army.seen,
                              home_threat=self.army.home_threat, radar_home=self.ctx.radar_home,
                              camera=self.ctx.camera),
                    match=dict(self.match), intel=self.intel.summary(), units=self.units_view(),
                    counts=dict(self.executor.counts) if self.executor else None, apm=round(self.apm(), 1),
                    timings_ms=avg, events=list(self.world.events)[-12:],
                    pointer=dict(position=getattr(self.pointer, 'position', None),
                                 gain=getattr(self.pointer, 'gain', None)) if self.pointer else None)

    def close(self):
        if self.runner:
            self.runner.cancel_all('closing')
        if self.watch:
            self.watch.close()
        if self.broker:
            self.broker.close()
        self.view.close()
        self.scene.close()
        (self.output / 'summary.json').write_text(json.dumps(self.view_state(), indent=2, default=str),
                                                  encoding='utf-8')
        for log in (self.events, self.decisions, self.actions):
            log.close()
