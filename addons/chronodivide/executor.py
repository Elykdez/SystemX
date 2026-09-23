"""Input owner: guarded, paced transactions with expectations and earned trust.

Skills describe abstract steps in canvas coordinates. The executor alone turns
them into desktop input, after checking the run generation, window ownership,
geometry, safe click zones, blocked keys, IME state, and user takeover. Tasks
(see tasks.py) own expectations; Trust records which skills have earned
optimistic verification.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time

from desktop.target import DesktopPause
from desktop import input as di
from .tasks import TaskFailed


class NotPlaying(DesktopPause):
    """The match is paused, in a menu, or not holding the pointer: wait, do not act."""


@dataclass(frozen=True)
class Step:
    kind: str                      # key | chord | click | move | nudge | wait | route
    keys: tuple = ()
    point: tuple | None = None     # canvas pixels
    button: str | None = None
    ms: float = 0.0
    delta: tuple | None = None
    points: tuple = ()            # finite waypoint route; point is the browser's mode button

    def __post_init__(self):
        if self.kind == 'route':
            if not 1 <= len(self.points) <= 16:
                raise ValueError('A waypoint route needs 1 to 16 points.')
            if self.keys not in ((), ('ctrl', 'shift')):
                raise ValueError('A waypoint route supports only move or attack-move.')

    @staticmethod
    def key(name):
        return Step('key', (name,))

    @staticmethod
    def repair_mode():
        return Step('repair_mode', ('k',))

    @staticmethod
    def chord(*names):
        return Step('chord', tuple(names))

    @staticmethod
    def click(point, button='left', modifiers=()):
        return Step('click', tuple(modifiers), (round(point[0]), round(point[1])), button)

    @staticmethod
    def move(point):
        return Step('move', (), (round(point[0]), round(point[1])))

    @staticmethod
    def wait(ms):
        return Step('wait', ms=ms)

    @staticmethod
    def nudge(dx, dy):
        """Raw pointer motion in desktop pixels (pointer calibration only)."""
        return Step('nudge', delta=(int(dx), int(dy)))

    @staticmethod
    def route(points, button_point, attack_move=False):
        """Queue a finite route; the browser uses a button, native input holds Z."""
        return Step('route', ('ctrl', 'shift') if attack_move else (),
                    (round(button_point[0]), round(button_point[1])),
                    points=tuple((round(p[0]), round(p[1])) for p in points))


class Trust:
    """Per-skill evidence: strict until enough clean live successes, demoted on any wrong effect."""

    def __init__(self, path=None, promote_after=50):
        self.path, self.promote_after = (Path(path) if path else None), promote_after
        self.records = {}
        if self.path and self.path.exists():
            self.records = json.loads(self.path.read_text(encoding='utf-8'))

    def level(self, skill):
        record = self.records.get(skill, {})
        return 'optimistic' if record.get('streak', 0) >= self.promote_after else 'strict'

    def update(self, skill, ok):
        record = self.records.setdefault(skill, dict(successes=0, failures=0, streak=0))
        if ok:
            record['successes'] += 1
            record['streak'] += 1
        else:
            record['failures'] += 1
            record['streak'] = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.records, indent=2), encoding='utf-8')


# Game keys the agent may press (bindings verified in Options > Keyboard, see profiles/keymap.json).
# Ordinary keys exclude D (repack), L/K (sell/repair), A (alliance), Delete, chat and diplomacy.
# Verified repair tasks use a dedicated repair_mode step for K.
SAFE_KEYS = {'q', 'w', 'e', 'r', 'h', 't', 's', 'g', 'x', 'n', 'm', 'p', 'space', *'123456789'}
TEAM_KEYS = '123456789'


def key_allowed(keys):
    modifiers = {k for k in keys if k in ('ctrl', 'shift', 'alt')}
    main = [k for k in keys if k not in ('ctrl', 'shift', 'alt')]
    if len(main) != 1 or 'alt' in modifiers or len(modifiers) > 1:
        # Ctrl+W/T/N close or open tabs; Alt focuses the browser menu.
        return False
    if modifiers:
        # Team keys only: Ctrl+N creates team N, Shift+N adds team N to the selection. Chrome
        # passes Ctrl+digit to the page first (tab switching is not a reserved shortcut) and the
        # game takes it in a windowed browser too (checked by hand, 2026-09-26). With several
        # tabs open a stray Ctrl+N could still switch tabs; the playing check then pauses.
        return main[0] in TEAM_KEYS
    return main[0] in SAFE_KEYS


class Guard:
    """Checks that must pass before any input event. Raises DesktopPause with a reason."""

    # Browser/OS shortcuts that could close, reload, or leave the game.
    BLOCKED = {('f5',), ('f11',), ('f12',), ('ctrl', 'w'), ('ctrl', 't'), ('ctrl', 'n'), ('ctrl', 'r'),
               ('ctrl', 'l'), ('ctrl', 'q'), ('ctrl', 'shift', 'w'), ('alt', 'f4'), ('ctrl', 'f4'),
               ('ctrl', 'tab'), ('alt', 'left'), ('alt', 'right'), ('backspace',), ('esc',)}

    def __init__(self, target, view, gate, generation, blocked=(), edge_margin=14,
                 takeover=None, check_ime=True, foreground=None, playing=None, deploy_allowed=None):
        self.target, self.view, self.gate, self.generation = target, view, gate, generation
        self.playing = playing           # callable -> (ok, reason); None disables the check
        # D deploys the MCV, but on a construction yard it packs the base up ("MCV Repacks"):
        # allowed only while this callable says no construction yard exists yet.
        self.deploy_allowed = deploy_allowed
        self.foreground = foreground or di.user32.GetForegroundWindow
        self.blocked = self.BLOCKED | {tuple(b) for b in blocked}
        self.edge_margin = edge_margin
        self.takeover = takeover
        self.check_ime = check_ime
        self._ime = (0.0, False)

    def ime_native(self):
        checked, native = self._ime
        if time.monotonic() - checked > 1.0:
            from desktop.target import ime_state
            native = ime_state(self.target.hwnd)['native']
            self._ime = (time.monotonic(), native)
        return native

    def before(self, steps):
        if not self.gate.allowed(self.generation):
            raise DesktopPause('Input cancelled (paused or stopped).')
        if self.takeover is not None and self.takeover():
            raise DesktopPause('You used the mouse or keyboard: the player paused. Resume with F8.')
        if self.target.geometry() != self.view.geometry:
            raise DesktopPause('Game window moved or resized. Re-locating before input.')
        if self.foreground() != self.target.hwnd:
            raise NotPlaying('Game window is not in front.')
        if self.playing is not None:
            ok, reason = self.playing()
            if not ok:
                raise NotPlaying(f'Not acting: {reason}.')
        letters = False
        for step in steps:
            if step.kind == 'repair_mode':
                letters = True
                if step.keys != ('k',) or ('k',) in self.blocked:
                    raise DesktopPause('Repair key is blocked by the safety list.')
            if step.kind == 'route':
                letters = True
                if ('z',) in self.blocked:
                    raise DesktopPause('Waypoint key Z is blocked by the safety list.')
                for point in step.points:
                    self.check_point(point)
            if step.kind in ('key', 'chord'):
                deploy = tuple(step.keys) == ('d',) and self.deploy_allowed is not None and self.deploy_allowed()
                if not deploy and (tuple(step.keys) in self.blocked or any((k,) in self.blocked for k in step.keys)
                                   or not key_allowed(step.keys)):
                    raise DesktopPause(f'Key {step.keys} is blocked by the safety list.')
                letters |= any(len(k) == 1 and k.isalpha() for k in step.keys)
            if step.point is not None and step.kind != 'route':
                self.check_point(step.point)
        if letters and self.check_ime and self.ime_native():
            raise DesktopPause('Chinese/native IME mode would swallow hotkeys. Switch input to English and resume.')

    def check_point(self, point):
        canvas = self.view.canvas
        x, y = point
        m = self.edge_margin
        if not (m <= x < canvas.width - m and m <= y < canvas.height - m):
            raise DesktopPause(f'Click {point} is outside the safe canvas zone.')
        g = self.view.geometry
        self.target.check_point(g.x + canvas.x + x, g.y + canvas.y + y, foreground=True)

    def desktop_point(self, point):
        g, canvas = self.view.geometry, self.view.canvas
        return g.x + canvas.x + point[0], g.y + canvas.y + point[1]


class AbsolutePointer:
    """Ordinary pointer: absolute SendInput positions map straight to the game."""

    name = 'absolute'
    position = None

    def __init__(self, guard):
        self.guard = guard

    def move_events(self, point):
        return di.mouse_events(self.guard.desktop_point(point), di.virtual_desktop())

    def move(self, point, send, sleep):
        send(self.move_events(point))

    def jump(self, dx, dy):
        x, y = di.cursor()
        return di.mouse_events((x + dx, y + dy), di.virtual_desktop()), (dx, dy)

    def nudge(self, dx, dy, send, sleep):
        events, _ = self.jump(dx, dy)
        send(events)

    def observe(self, position):
        pass

    def resync(self):
        return []


class LockedPointer:
    """Browser pointer lock: the game draws its own cursor and reads relative motion.

    Chrono Divide profiles send relative input; calibration measures its gain.
    The older absolute path supports clients that re-center the hidden OS cursor after
    each move. An absolute jump from the *current* OS position arrives as exactly
    that delta, free of Windows acceleration. One jump is sent per batch because
    re-centering happens between events. The game cursor position is tracked in
    canvas pixels and corrected from vision (the drawn cursor sprite).
    """

    name = 'locked'

    def __init__(self, guard, gain=1.0, cursor=None, clip=None, relative=False):
        self.guard, self.gain = guard, gain
        self.position = None                 # game cursor, canvas pixels (None = unknown)
        self.cursor = cursor or (lambda: di.cursor())
        self.clip = clip                     # callable -> (left, top, right, bottom) or None
        self.relative = relative

    def observe(self, position):
        """Vision found the cursor sprite: trust it over dead reckoning."""
        self.position = tuple(position)

    def jump(self, dx, dy):
        """One relative event, or an absolute jump from the current OS cursor."""
        if self.relative:
            # Chrome can leave an injected absolute cursor pinned at the window edge under lock.
            return [di.relative_move_event(dx, dy)], (dx, dy)
        x, y = self.cursor()
        left, top, right, bottom = self.clip() if self.clip else (x - 10**4, y - 10**4, x + 10**4, y + 10**4)
        tx = max(left + 1, min(right - 2, x + dx))
        ty = max(top + 1, min(bottom - 2, y + dy))
        return di.mouse_events((tx, ty), di.virtual_desktop()), (tx - x, ty - y)

    def move_events(self, point):
        """One jump toward a canvas point (as far as the clip allows; tracking follows what was sent)."""
        if self.position is None:
            # Only this task fails; the reflex calibrates next (a pause here stopped the agent for
            # the rest of live game 28: nothing resumes a pause that no person caused).
            raise TaskFailed('Game cursor position unknown; calibrating the pointer first.')
        dx = round((point[0] - self.position[0]) * self.gain)
        dy = round((point[1] - self.position[1]) * self.gain)
        events, (sent_x, sent_y) = self.jump(dx, dy)
        self.position = (self.position[0] + sent_x / self.gain, self.position[1] + sent_y / self.gain)
        return events

    def nudge(self, dx, dy, send, sleep, step=120, pause=0.012, tries=40):
        """Deliver the full calibration delta, allowing browser recentering between jumps.

        A single jump can be clipped by the OS window. Measuring gain against its requested
        distance then mistakes lost motion for acceleration and blocks every subsequent click.
        """
        for _ in range(tries):
            if dx == 0 and dy == 0:
                return
            self.guard.before([])
            scale = min(1.0, step / max(abs(dx), abs(dy)))
            events, (sent_x, sent_y) = self.jump(round(dx * scale), round(dy * scale))
            send(events)
            dx, dy = dx - sent_x, dy - sent_y
            sleep(pause)
        if dx or dy:
            raise TaskFailed(f'Calibration motion was clipped ({dx}, {dy} pixels remaining).')

    def move(self, point, send, sleep, step=120, pause=0.012, tries=40):
        """Move the game cursor to a canvas point in steps of at most `step` desktop pixels.

        Under pointer lock Chrome keeps the hidden OS cursor inside its window and pulls it back to
        the window's center only once it enters the outer band of the window (15% of its size);
        until then it stays where the last move left it. One long move from there ran into the
        window's edge and fell short, so the click after it landed short too (live games 29-31:
        radar clicks 40-60 px low, every camera jump failed). Steps smaller than that band never
        reach the edge, and the pause lets Chrome pull the cursor back between them."""
        if self.position is None:
            raise TaskFailed('Game cursor position unknown; calibrating the pointer first.')
        for _ in range(tries):
            dx = round((point[0] - self.position[0]) * self.gain)
            dy = round((point[1] - self.position[1]) * self.gain)
            if abs(dx) <= 1 and abs(dy) <= 1:
                return
            scale = min(1.0, step / max(abs(dx), abs(dy)))
            events, (sent_x, sent_y) = self.jump(round(dx * scale), round(dy * scale))
            send(events)
            self.position = (self.position[0] + sent_x / self.gain, self.position[1] + sent_y / self.gain)
            sleep(pause)
        raise TaskFailed(f'The pointer did not reach {point} (stuck at {self.position}).')

    def resync(self):
        self.position = None
        return []


class Executor:
    """Sends guarded, paced input for tasks; counts transactions, events, and game commands."""

    def __init__(self, guard, pointer, sender=di.send, clock=time.monotonic, sleep=time.sleep,
                 step_gap_ms=12.0, log=None, click_settle_ms=60.0):
        self.guard, self.pointer = guard, pointer
        self.sender, self.clock, self.sleep = sender, clock, sleep
        self.step_gap = step_gap_ms / 1000
        self.click_settle = click_settle_ms / 1000
        self.log = log
        self.last_input = 0.0
        self.last_click = -1e9
        self.counts = dict(transactions=0, events=0, commands=0)

    def _send(self, events):
        if not events:
            return
        sent = self.sender(events)
        self.counts['events'] += sent
        if sent != len(events):
            di.release_all()
            raise DesktopPause('Windows rejected part of an input batch. Input stopped.')

    def _events(self, step):
        if step.kind in ('key', 'chord', 'repair_mode'):
            return di.chord_events(step.keys)
        if step.kind == 'move':
            return self.pointer.move_events(step.point)
        if step.kind == 'click':
            mods = list(step.keys)
            return ([di.key_event(k) for k in mods] + self.pointer.move_events(step.point) +
                    di.mouse_events(None, None, step.button, move=False) +
                    [di.key_event(k, up=True) for k in reversed(mods)])
        raise ValueError(step.kind)

    def _modified_click(self, step, hold_s=0.04):
        """Ctrl/Shift+click in stages: the game reads modifiers on its own frame, so keys sent in
        the same instant as the click were already up when it looked (live game 4: every
        Ctrl+Shift attack-move was ignored while plain clicks worked)."""
        mods = list(step.keys)
        self._send([di.key_event(k) for k in mods])
        self.sleep(hold_s)
        self.pointer.move(step.point, self._send, self.sleep)
        self.sleep(hold_s)
        self._send(di.mouse_events(None, None, step.button, move=False))
        self.sleep(hold_s)
        self._send([di.key_event(k, up=True) for k in reversed(mods)])

    def _hover_click(self, step, rest_s=0.05):
        """Move, let the game see what is under the pointer, then press: a click sent with its move
        was read against the previous hover (live game 23: plain clicks on an enemy building never
        became attack orders, for eight minutes, while Ctrl+Shift attack-moves worked)."""
        self.pointer.move(step.point, self._send, self.sleep)
        self.sleep(rest_s)
        self._send(di.mouse_events(None, None, step.button, move=False))

    def _modified_chord(self, step, hold_s=0.04):
        """Ctrl/Shift+key in stages, for the same reason as `_modified_click` (Ctrl+1 creates a team)."""
        mods = [k for k in step.keys if k in di.MODIFIERS]
        main = [k for k in step.keys if k not in di.MODIFIERS]
        self._send([di.key_event(k) for k in mods])
        self.sleep(hold_s)
        self._send(di.chord_events(main))
        self.sleep(hold_s)
        self._send([di.key_event(k, up=True) for k in reversed(mods)])

    def _route(self, step):
        """Release Z even on takeover; only key-up cleanup may bypass the input guard.

        Exiting commits any partial draft. A subsequent ordinary order replaces that queue.
        """
        try:
            self._send(di.chord_events(('z',)))   # finish any mode left active by the UI button
            self.sleep(0.04)
            self.guard.before([])
            self._send([di.key_event('z')])
            self.sleep(0.04)
            for point in step.points:
                click = Step.click(point, modifiers=step.keys)
                self.guard.before([click])
                settle = self.click_settle - (self.clock() - self.last_click)
                if settle > 0:
                    self.sleep(settle)
                self.guard.before([click])
                self.pointer.move(point, self._send, self.sleep)
                self.sleep(0.05)
                self.guard.before([click])
                if step.keys:
                    self._send([di.key_event(k) for k in step.keys])
                    self.sleep(0.04)
                    self.guard.before([click])
                self._send(di.mouse_events(None, None, 'left', move=False))
                self.sleep(0.04)
                if step.keys:
                    self._send([di.key_event(k, up=True) for k in reversed(step.keys)])
                self.last_click = self.clock()
        finally:
            self._send([di.key_event('z', up=True)])

    def act(self, name, lane, steps, commands=1):
        """Run one transaction now; every step is re-guarded and paced."""
        steps = list(steps)
        if any(s.kind == 'repair_mode' for s in steps) and not name.startswith('repair:'):
            raise DesktopPause('Repair mode is restricted to verified repair tasks.')
        self.guard.before(steps)
        started = self.clock()
        try:
            for index, step in enumerate(steps):
                if step.kind == 'wait':
                    self.sleep(step.ms / 1000)
                    continue
                gap = self.step_gap - (self.clock() - self.last_input)
                if gap > 0:
                    self.sleep(gap)
                if index:
                    self.guard.before([step])
                if step.kind in ('move', 'click', 'nudge'):
                    # The game reads the buttons once per frame: motion in the frame of a click's
                    # release turned a right click into a right-drag, which scrolls the map (live
                    # games 15 and 16: the cursor stayed at the deselect point while the camera slid
                    # into the shroud, and every later click missed).
                    settle = self.click_settle - (self.clock() - self.last_click)
                    if settle > 0:
                        self.sleep(settle)
                if step.kind == 'route':
                    self._route(step)
                elif step.kind == 'click' and step.keys:
                    self._modified_click(step)
                elif step.kind == 'click':
                    self._hover_click(step)
                elif step.kind == 'chord' and any(k in di.MODIFIERS for k in step.keys):
                    self._modified_chord(step)
                elif step.kind == 'move':
                    self.pointer.move(step.point, self._send, self.sleep)
                elif step.kind == 'nudge':
                    self.pointer.nudge(*step.delta, self._send, self.sleep)
                else:
                    self._send(self._events(step))
                self.last_input = self.clock()
                if step.kind == 'click':
                    self.last_click = self.last_input
        except BaseException:
            # Each step is one atomic batch with its own key-ups, so only modifiers can matter.
            di.release_all(buttons=())
            if hasattr(self.pointer, 'position'):
                self.pointer.position = None
            raise
        self.counts['transactions'] += 1
        self.counts['commands'] += commands
        elapsed = round((self.clock() - started) * 1000, 2)
        if self.log:
            self.log(dict(time=time.time(), act=name, lane=lane, steps=len(steps), input_ms=elapsed,
                          points=[s.point for s in steps if s.point is not None],
                          keys=[s.keys for s in steps if s.keys]))
        return elapsed
