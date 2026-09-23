"""Find the window running the game and, for a game drawn in a page (Chrono Divide in a browser),
the canvas inside it. Which window: calibration.py ranks them by the client profile."""

import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import dataclass
import subprocess
import time

import numpy as np

from desktop.target import DesktopPause, Target, kernel32, user32, windows

user32.AttachThreadInput.argtypes = [wt.DWORD, wt.DWORD, wt.BOOL]
user32.BringWindowToTop.argtypes = [wt.HWND]
user32.ShowWindow.argtypes = [wt.HWND, ct.c_int]

TITLE = 'Chrono Divide'
GAME_URL = 'https://game.chronodivide.com/'
MIN_CANVAS = (640, 480)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def contains(self, x, y):
        return self.x <= x < self.right and self.y <= y < self.bottom

    def offset(self, dx, dy):
        return Rect(self.x + dx, self.y + dy, self.width, self.height)

    def tuple(self):
        return self.x, self.y, self.width, self.height


def find_game_window(profile=None, setup=None):
    """The game window of a client profile (Chrono Divide by default): minimized copies are ignored;
    among several, the one in front wins (calibration.pick_window)."""
    from .calibration import pick_window
    from .client import ClientProfile
    profile = profile if profile is not None else ClientProfile.load()
    return pick_window(profile, setup, [t for t in windows() if not user32.IsIconic(t.hwnd)])


def ensure_game(timeout=45.0, launch=True, sleep=time.sleep, clock=time.monotonic, opener=None, profile=None):
    """Unattended runs: the game window, restored if minimized; if no window shows the game,
    it is opened in Chrome (the menus are then clicked through by the autoplay pilot)."""
    from .calibration import rank_windows
    from .client import ClientProfile
    profile = profile if profile is not None else ClientProfile.load()

    def games():
        return [c.target for c in rank_windows(profile, windows()) if c.titled]
    found = games()
    if not found and launch:
        (opener or _open_in_chrome)(GAME_URL)
        deadline = clock() + timeout
        while not found and clock() < deadline:
            sleep(1.0)
            found = games()
    if not found:
        raise DesktopPause(f'No window shows the game; open {GAME_URL} in Chrome.')
    for target in found:
        if user32.IsIconic(target.hwnd):
            bring_to_front(target)             # restores it
    return find_game_window(profile)


def _open_in_chrome(url):
    subprocess.Popen(['cmd', '/c', 'start', '', 'chrome', url], creationflags=0x08000000)   # no console


def bring_to_front(target):
    """Make the game window the foreground window (unattended runs only). True on success.

    Windows lets only the foreground thread change the foreground window, so the call
    briefly shares that thread's input state.
    """
    current = kernel32.GetCurrentThreadId()
    thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    attached = bool(thread) and thread != current and bool(user32.AttachThreadInput(current, thread, True))
    try:
        if user32.IsIconic(target.hwnd):
            user32.ShowWindow(target.hwnd, 9)            # SW_RESTORE
        user32.BringWindowToTop(target.hwnd)
        user32.SetForegroundWindow(target.hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(current, thread, False)
    return user32.GetForegroundWindow() == target.hwnd


CANVAS_SIZE = (1024, 768)


def _mode(pixels):
    colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def locate_canvas(client, size=CANVAS_SIZE, max_ties=25):
    """Canvas rectangle in client pixels: the one size-sized block with no page background.

    The page margin is a flat gray (its exact value differs per browser). Browser chrome,
    info bars, and the glow a browser draws around an automated page are all outside the
    canvas, so the canvas is found as the window of known size that contains no background
    pixel, using a summed-area table over the non-background mask.

    Which color is the page background is not known in advance: the window's left and right
    edges can be the frame or a browser side panel (the Claude panel is darker than the page).
    Each candidate (both edge colors and the most common non-black color) is tried; the real
    background pins the canvas down to one spot, while a wrong one lets the block slide over
    plain page, so the candidate with the fewest fitting positions wins. The canvas sits where the
    block holds the most content; when many places tie for that (more than `max_ties` within 0.2%),
    the page gives no answer: live 2026-09-28, a dark page around the canvas tied it in 2238 places
    (pinned canvases tie in about 7) and the first of them was taken. The caller must then know
    better (recorder.GameView: the HUD anchors, or the place found before).
    """
    height, width = client.shape[:2]
    cw, ch = size
    if width < cw or height < ch:
        raise DesktopPause(f'The browser area ({width}x{height}) is smaller than the {cw}x{ch} game. '
                           'Make the window larger.')
    # The page background shows at the outer edges beside the centered canvas, however
    # narrow the margin (a mostly black menu screen must not be mistaken for background).
    rows = slice(int(height * .35), int(height * .9), 5)
    margin = max(1, min(8, (width - cw) // 2 - 2))
    sample = client[::4, ::4].reshape(-1, 3)
    lit = sample[sample.astype(np.int16).sum(axis=1) >= 30]
    candidates = [_mode(client[rows, 1:1 + margin]), _mode(client[rows, width - 1 - margin:width - 1])]
    if len(lit):
        candidates.append(_mode(lit))
    best = None
    for background in {tuple(int(v) for v in c): c for c in candidates}.values():
        content = np.any(client != background, axis=2).astype(np.int32)
        table = np.zeros((height + 1, width + 1), np.int64)
        table[1:, 1:] = content.cumsum(0).cumsum(1)
        sums = table[ch:, cw:] - table[:-ch, cw:] - table[ch:, :-cw] + table[:-ch, :-cw]
        fits = int(np.count_nonzero(sums >= 0.97 * cw * ch))
        if not fits:
            continue
        top = int(sums.max())
        ties = int(np.count_nonzero(sums >= top - 0.002 * cw * ch))
        y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
        if best is None or fits < best[0]:
            best = (fits, int(x), int(y), ties)
    if best is None or best[3] > max_ties:
        raise DesktopPause('Cannot find the game canvas. Keep the game visible at 100% zoom, '
                           'with the browser window larger than the game.')
    return Rect(best[1], best[2], cw, ch)


class CURSORINFO(ct.Structure):
    _fields_ = [('cbSize', wt.DWORD), ('flags', wt.DWORD), ('hCursor', wt.HANDLE), ('ptScreenPos', wt.POINT)]


user32.GetCursorInfo.argtypes = [ct.POINTER(CURSORINFO)]
user32.GetClipCursor.argtypes = [ct.POINTER(wt.RECT)]


def pointer_state():
    """Read-only evidence of browser pointer lock: hidden and/or clipped OS cursor."""
    info = CURSORINFO(ct.sizeof(CURSORINFO))
    clip = wt.RECT()
    showing = bool(user32.GetCursorInfo(ct.byref(info)) and info.flags & 1)
    user32.GetClipCursor(ct.byref(clip))
    screen = tuple(user32.GetSystemMetrics(i) for i in (76, 77, 78, 79))
    clipped = (clip.left, clip.top, clip.right - clip.left, clip.bottom - clip.top) != screen
    return dict(showing=showing, clipped=clipped, clip=[clip.left, clip.top, clip.right, clip.bottom],
                position=[info.ptScreenPos.x, info.ptScreenPos.y], foreground=user32.GetForegroundWindow() or 0)


__all__ = ['Rect', 'Target', 'bring_to_front', 'ensure_game', 'find_game_window', 'locate_canvas',
           'pointer_state']
