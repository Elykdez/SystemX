"""Cancellable, foreground-only Windows mouse input."""

import ctypes as ct
from ctypes import wintypes as wt
import threading
import time

from .target import user32, DesktopPause


class Gate:
    def __init__(self):
        self.lock = threading.RLock()
        self.generation = 0
        self.running = False
        self.single = False
        self.closed = False

    def start(self, single=False):
        with self.lock:
            if self.closed:
                return
            self.generation += 1
            self.running, self.single = True, single

    def pause(self):
        with self.lock:
            self.generation += 1
            self.running = self.single = False

    def stop(self):
        with self.lock:
            self.pause()
            self.closed = True

    def allowed(self, generation):
        with self.lock:
            return self.running and not self.closed and generation == self.generation


class MouseInput(ct.Structure):
    _fields_ = [('dx', wt.LONG), ('dy', wt.LONG), ('mouseData', wt.DWORD),
                ('dwFlags', wt.DWORD), ('time', wt.DWORD), ('dwExtraInfo', ct.c_size_t)]


class KeybdInput(ct.Structure):
    _fields_ = [('wVk', wt.WORD), ('wScan', wt.WORD), ('dwFlags', wt.DWORD),
                ('time', wt.DWORD), ('dwExtraInfo', ct.c_size_t)]


class InputUnion(ct.Union):
    _fields_ = [('mi', MouseInput), ('ki', KeybdInput)]


class Input(ct.Structure):
    _fields_ = [('type', wt.DWORD), ('data', InputUnion)]


user32.SendInput.argtypes = [wt.UINT, ct.POINTER(Input), ct.c_int]
user32.SendInput.restype = wt.UINT
user32.MapVirtualKeyW.argtypes = [wt.UINT, wt.UINT]
user32.MapVirtualKeyW.restype = wt.UINT

VK = {'backspace': 0x08, 'tab': 0x09, 'enter': 0x0D, 'shift': 0x10, 'ctrl': 0x11, 'alt': 0x12,
      'pause': 0x13, 'esc': 0x1B, 'space': 0x20, 'pageup': 0x21, 'pagedown': 0x22, 'end': 0x23,
      'home': 0x24, 'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28, 'insert': 0x2D, 'delete': 0x2E,
      **{str(d): 0x30 + d for d in range(10)}, **{chr(c): c - 32 for c in range(ord('a'), ord('z') + 1)},
      **{f'num{d}': 0x60 + d for d in range(10)}, **{f'f{n}': 0x6F + n for n in range(1, 13)}}
EXTENDED = {'pageup', 'pagedown', 'end', 'home', 'left', 'up', 'right', 'down', 'insert', 'delete'}
MODIFIERS = ('ctrl', 'shift', 'alt')


def key_event(name, up=False):
    """One keyboard INPUT with virtual key plus scan code; browsers read both."""
    if name not in VK:
        raise ValueError(f'Unsupported key: {name}')
    vk = VK[name]
    flags = (0x0001 if name in EXTENDED else 0) | (0x0002 if up else 0)
    return Input(1, InputUnion(ki=KeybdInput(vk, user32.MapVirtualKeyW(vk, 0), flags, 0, 0)))


def chord_events(keys):
    """Press modifiers first and release in reverse order: ('ctrl', '1') -> 4 events."""
    keys = list(keys)
    return [key_event(k) for k in keys] + [key_event(k, up=True) for k in reversed(keys)]


def mouse_events(point, desktop, button=None, move=True):
    """Absolute move to a desktop point and an optional paired button click."""
    events = []
    if move:
        dx, dy = absolute_point(*point, desktop)
        events.append(Input(0, InputUnion(MouseInput(dx, dy, 0, 0xC001, 0, 0))))
    if button is not None:
        down, up = {'left': (2, 4), 'right': (8, 16), 'middle': (32, 64)}[button]
        events += [Input(0, InputUnion(MouseInput(0, 0, 0, down, 0, 0))),
                   Input(0, InputUnion(MouseInput(0, 0, 0, up, 0, 0)))]
    return events


def relative_move_event(dx, dy):
    return Input(0, InputUnion(MouseInput(int(dx), int(dy), 0, 0x0001, 0, 0)))


def send(events):
    """Send a batch; returns the number Windows accepted."""
    if not events:
        return 0
    batch = (Input * len(events))(*events)
    return user32.SendInput(len(events), batch, ct.sizeof(Input))


def release_all(keys=MODIFIERS, buttons=('left', 'right', 'middle')):
    """Best-effort cleanup after cancellation: key-ups and button-ups are harmless when not held."""
    events = [key_event(k, up=True) for k in keys]
    events += [Input(0, InputUnion(MouseInput(0, 0, 0, {'left': 4, 'right': 16, 'middle': 64}[b], 0, 0)))
               for b in buttons]
    return send(events)


def virtual_desktop():
    return tuple(user32.GetSystemMetrics(i) for i in (76, 77, 78, 79))


def absolute_point(x, y, desktop):
    left, top, width, height = desktop
    if width < 2 or height < 2 or not (left <= x < left + width and top <= y < top + height):
        raise DesktopPause("Click is outside the virtual desktop.")
    return round((x - left) * 65535 / (width - 1)), round((y - top) * 65535 / (height - 1))


def cursor():
    point = wt.POINT()
    if not user32.GetCursorPos(ct.byref(point)):
        raise DesktopPause("Cannot read pointer position.")
    return point.x, point.y


def clip_rect():
    rect = wt.RECT()
    user32.GetClipCursor(ct.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


class Mouse:
    def __init__(self, target, gate):
        self.target, self.gate = target, gate
        self.expected_cursor = cursor()

    def move(self, x, y, geometry, generation):
        with self.gate.lock:
            if not self.gate.allowed(generation):
                raise DesktopPause("Pointer move cancelled.")
            if self.target.geometry() != geometry:
                raise DesktopPause("Window geometry changed before pointer movement.")
            self.target.check_point(x, y, foreground=True)
            if cursor() != self.expected_cursor:
                raise DesktopPause("Pointer moved. Resume when ready.")
            if any(user32.GetAsyncKeyState(key) & 0x8000 for key in (1, 2, 4, 16, 17, 18)):
                raise DesktopPause("Release buttons and modifier keys before resuming.")
            desktop = tuple(user32.GetSystemMetrics(i) for i in (76, 77, 78, 79))
            dx, dy = absolute_point(x, y, desktop)
            event = Input(0, InputUnion(MouseInput(dx, dy, 0, 0xC001, 0, 0)))
            if user32.SendInput(1, ct.byref(event), ct.sizeof(Input)) != 1:
                raise DesktopPause("Windows rejected pointer movement.")
            self.expected_cursor = cursor()

    def click(self, x, y, button, geometry, generation, park=None):
        if button not in ('left', 'right'):
            raise ValueError("Only left/right clicks are supported.")
        self.move(x, y, geometry, generation)
        # Frame-polled games need movement, button edges, and parking in separate frames.
        time.sleep(.05)
        down, up = (2, 4) if button == 'left' else (8, 16)
        with self.gate.lock:
            if not self.gate.allowed(generation):
                raise DesktopPause("Action cancelled.")
            if self.target.geometry() != geometry:
                raise DesktopPause("Window geometry changed before input.")
            self.target.check_point(x, y, foreground=True)
            if cursor() != self.expected_cursor:
                raise DesktopPause("Pointer moved. Focus the game and resume when ready.")
            if any(user32.GetAsyncKeyState(key) & 0x8000 for key in (1, 2, 4, 16, 17, 18)):
                raise DesktopPause("Release mouse buttons and modifier keys before resuming.")
            pressed = Input(0, InputUnion(MouseInput(0, 0, 0, down, 0, 0)))
            sent = user32.SendInput(1, ct.byref(pressed), ct.sizeof(Input))
        try:
            if sent != 1:
                raise DesktopPause("Windows did not accept the complete click. Input stopped.")
            time.sleep(.05)
        finally:
            released = Input(0, InputUnion(MouseInput(0, 0, 0, up, 0, 0)))
            if user32.SendInput(1, ct.byref(released), ct.sizeof(Input)) != 1:
                raise DesktopPause("Windows rejected button release. Input stopped.")
        time.sleep(.05)
        if park is not None:
            self.move(*park, geometry, generation)
            # The player's visual postcondition waits for cursor/board animation.


class Hotkeys:
    """A dedicated Windows message thread keeps stop responsive during inference."""
    def __init__(self, callback):
        self.callback = callback
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(3):
            raise RuntimeError("Hotkey registration timed out.")
        if self.error:
            raise RuntimeError(self.error)

    def _run(self):
        self.thread_id = ct.windll.kernel32.GetCurrentThreadId()
        registered = []
        try:
            for number, key in enumerate((0x77, 0x78, 0x79), 1):
                if not user32.RegisterHotKey(None, number, 0x4000, key):
                    self.error = "F8/F9/F10 hotkeys are in use. Close the other player and retry."
                    return
                registered.append(number)
            self.ready.set()
            message = wt.MSG()
            while user32.GetMessageW(ct.byref(message), None, 0, 0) > 0:
                if message.message == 0x0312:
                    self.callback({1: 'run', 2: 'step', 3: 'pause'}[message.wParam])
        finally:
            for number in registered:
                user32.UnregisterHotKey(None, number)
            self.ready.set()

    def close(self):
        user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
        self.thread.join(timeout=2)
