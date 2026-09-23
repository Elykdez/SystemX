"""Find the game: which window shows it, and where its view is inside that window.

The Minesweeper player's calibration applied to RA2: pick a window, capture it, and find the
view automatically, or mark it by hand in the setup preview (setup_ui.py). Windows are ranked by
the client profile's title hints and processes; File Explorer and other shell windows never
count, even for a folder named after the game. The view is found from the sidebar's HUD
anchors, which every RA2 client draws at 1:1: the bottom anchor gives the view's right and
bottom edges, the top buttons its top edge. A game drawn in a page (a browser) is then pinned
on the page by its size (window.locate_canvas); a game that fills its window starts at the
window's left edge. The choice is saved (.cache/chronodivide/setup.json) for the next start.
"""

import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time

import cv2
import numpy as np

from desktop.target import DesktopPause, Target, kernel32, user32, windows
from .window import Rect, locate_canvas

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / '.cache' / 'chronodivide' / 'setup.json'
SHELL_PROCESSES = {'explorer.exe', 'searchhost.exe', 'shellexperiencehost.exe', 'textinputhost.exe'}
SHELL_CLASSES = {'CabinetWClass', 'ExploreWClass', 'Progman', 'WorkerW', 'Shell_TrayWnd'}
ANCHOR_SHARE = 0.7            # share of anchor pixels within the tolerance (as perception/modes.py)
TOLERANCE = 12
ASPECTS = (4 / 3, 16 / 10, 16 / 9, 5 / 4, 3 / 2)

kernel32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD)]
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ct.c_int]


def process_name(pid):
    """Executable name of a process (lower case), or '' when it cannot be read."""
    handle = kernel32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ''
    try:
        buffer, size = ct.create_unicode_buffer(1024), wt.DWORD(1024)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ct.byref(size)):
            return ''
        return Path(buffer.value).name.lower()
    finally:
        kernel32.CloseHandle(handle)


def class_name(hwnd):
    buffer = ct.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value


@dataclass(frozen=True)
class Candidate:
    target: Target
    process: str
    score: int
    titled: bool                 # the title names the game

    @property
    def label(self):
        return f'{self.target.title[:60]}  ({self.process or "?"}, {self.target.hwnd})'


def _titled(title, hints):
    title = title.lower()
    return any(h.lower() in title or h.lower().replace(' ', '') in title.replace(' ', '') for h in hints)


def rank_windows(profile, found=None, process=process_name, klass=class_name):
    """Windows that may show the game, best first: a title naming the game counts most, then a
    process the profile expects (a browser, the game's executable). Shell windows are left out."""
    ranked = []
    for target in (windows() if found is None else found):
        name = process(getattr(target, 'pid', 0)) or ''
        if name in SHELL_PROCESSES or klass(target.hwnd) in SHELL_CLASSES:
            continue
        titled = _titled(target.title, profile.titles)
        score = 4 * titled + 2 * (name in profile.processes)
        ranked.append(Candidate(target, name, score, titled))
    return sorted(ranked, key=lambda c: -c.score)


def pick_window(profile, setup=None, found=None, foreground=None, **kwargs):
    """The game window: the one saved in the setup while it lives, else the one whose title names
    the game (the one in front when there are several)."""
    if setup is not None and setup.window:
        saved = setup.window
        for target in (windows() if found is None else found):
            if (target.hwnd, target.pid, target.created) == (saved.get('hwnd'), saved.get('pid'), saved.get('created')):
                return target
    ranked = [c for c in rank_windows(profile, found, **kwargs) if c.titled]
    if not ranked:
        raise DesktopPause(f'No window shows {profile.name}. Open the game, or pick its window in Setup.')
    best = [c for c in ranked if c.score == ranked[0].score]
    if len(best) > 1:
        front = (foreground or user32.GetForegroundWindow)()
        best = [c for c in best if c.target.hwnd == front] or best
        if len(best) > 1:
            raise DesktopPause('Several windows show the game. Close the extra ones, or pick one in Setup.')
    return best[0].target


# -- the view inside the window ---------------------------------------------------------
@dataclass(frozen=True)
class Found:
    rect: Rect                   # the game view, in client pixels
    anchors: dict                # anchor name -> share of matching pixels

    @property
    def size(self):
        return self.rect.width, self.rect.height


def match_anchor(image, template, area=None):
    """Best place of a HUD anchor in image (within area (x, y, w, h)): ((x, y), share of its pixels
    within the tolerance), or None when it does not fit."""
    if template.ndim == 4:
        matches = [match_anchor(image, sample, area) for sample in template]
        return max((match for match in matches if match is not None), key=lambda match: match[1], default=None)
    ox, oy = 0, 0
    if area is not None:
        x, y, w, h = area
        ox, oy = max(0, x), max(0, y)
        image = image[oy:max(oy, y + h), ox:max(ox, x + w)]
    th, tw = template.shape[:2]
    if image.shape[0] < th or image.shape[1] < tw:
        return None
    scores = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    _, _, (x, y), _ = cv2.minMaxLoc(scores)
    crop = image[y:y + th, x:x + tw].astype(np.int16)
    share = float(np.mean(np.abs(crop - template.astype(np.int16)).max(axis=2) <= TOLERANCE))
    return (x + ox, y + oy), share


def _left_by_background(client, top, height, right):
    """A page's canvas starts right of the page background: the left edge where the first run of
    the window's leftmost color ends, looking at rows well inside the view. None: no margin."""
    rows = client[top + int(height * .35):top + int(height * .9):5]
    if not len(rows):
        return None
    edge = rows[:, 0].astype(np.int16)
    if np.ptp(edge, axis=0).max() > 3:
        return None                                  # the window's edge is not a flat page margin
    background = np.median(edge, axis=0)
    flat = np.all(np.abs(rows.astype(np.int16) - background).max(axis=2) <= 3, axis=0)
    columns = np.flatnonzero(~flat[:right])
    return int(columns[0]) if len(columns) else None


def find_game(client, profile, templates=None):
    """The game view in a captured window (BGR client pixels), from the sidebar's HUD anchors, or
    None when they are not on screen (menus, loading, another sidebar style, a stretched game)."""
    templates = templates if templates is not None else profile.hud_templates()
    spec = profile.sidebar['anchors']
    if 'sidebar_bottom' not in templates or 'top_buttons' not in templates:
        return None
    found = match_anchor(client, templates['sidebar_bottom'])
    if found is None or found[1] < ANCHOR_SHARE:
        return None
    (bx, by), bottom_share = found
    left_of_sidebar = bx - spec['sidebar_bottom'][0]
    bottom = by - spec['sidebar_bottom'][1]          # the anchor is that far above the bottom
    right = left_of_sidebar + profile.sidebar['width']
    tx, ty, tw, th = spec['top_buttons']
    top_match = match_anchor(client, templates['top_buttons'], (left_of_sidebar + tx - 3, 0, tw + 6, by))
    anchors = dict(sidebar_bottom=round(bottom_share, 3))
    if top_match is not None and top_match[1] >= ANCHOR_SHARE:
        top = top_match[0][1] - ty
        anchors['top_buttons'] = round(top_match[1], 3)
    else:
        top = bottom - profile.size[1]              # the game menu changes the top buttons
    height = bottom - top
    if top < 0 or height < 240 or right > client.shape[1] or bottom > client.shape[0]:
        return None
    if not profile.embedded:
        return Found(Rect(0, top, right, height), anchors)
    widths = []
    left = _left_by_background(client, top, height, left_of_sidebar)
    if left is not None:
        widths.append(right - left)
    if height == profile.size[1]:
        widths.append(profile.size[0])
    widths += [round(height * aspect) for aspect in ASPECTS]
    for width in dict.fromkeys(w for w in widths if 320 <= w <= right):
        try:
            rect = locate_canvas(client, (width, height))
        except DesktopPause:
            continue
        if abs(rect.right - right) <= 1 and abs(rect.y - top) <= 1:
            return Found(Rect(right - width, top, width, height), anchors)
    # The page does not pin the left edge down (a dark page around the canvas, live 2026-09-28): the
    # width the profile gives this height, else the page margin's, else 4:3. Setup shows it; a drag
    # corrects it.
    if height == profile.size[1]:
        width = profile.size[0]
    else:
        width = right - left if left is not None else round(height * 4 / 3)
    width = min(width, right)
    return Found(Rect(right - width, top, width, height), dict(anchors, width='assumed'))


# -- the saved choice ---------------------------------------------------------------------
@dataclass
class Setup:
    profile: str = 'chronodivide'                    # profile name (profiles/clients/) or path
    window: dict | None = None                       # hwnd, pid, created, title, process of the game window
    view: list | None = None                         # the view marked or found: [x, y, w, h] in client pixels
    client: list | None = None                       # the window's client size when the view was set
    saved: float = field(default_factory=time.time)

    @property
    def size(self):
        return tuple(self.view[2:]) if self.view else None

    def marked(self, geometry):
        """(Rect, client size) of the saved view, for a window of that client size; else None."""
        if not self.view or not self.client or [geometry.width, geometry.height] != list(self.client):
            return None
        return Rect(*self.view), tuple(self.client)

    def remember(self, target, process, rect, geometry):
        self.window = dict(hwnd=target.hwnd, pid=target.pid, created=target.created, title=target.title,
                           process=process)
        self.view = [rect.x, rect.y, rect.width, rect.height]
        self.client = [geometry.width, geometry.height]
        self.saved = time.time()


def open_view(profile=None, setup=None, **kwargs):
    """(profile, window, GameView) as set up: the setup's profile and window (else the best one by
    title), and its saved view while the window keeps its client size."""
    from .client import ClientProfile
    from .recorder import GameView
    from .window import find_game_window
    setup = setup if setup is not None else load_setup()
    profile = profile or ClientProfile.load(setup.profile if setup else 'chronodivide')
    target = find_game_window(profile, setup)
    marked = None
    if setup is not None and (setup.window or {}).get('hwnd') == target.hwnd:
        try:
            marked = setup.marked(target.geometry())
        except DesktopPause:
            pass
    size = (marked[0].width, marked[0].height) if marked else profile.size
    return profile, target, GameView(target, profile=profile, size=size, marked=marked, **kwargs)


def load_setup(path=SETUP):
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        return Setup(**{k: raw[k] for k in Setup.__dataclass_fields__ if k in raw})
    except (OSError, ValueError, TypeError):
        return None


def save_setup(setup, path=SETUP):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(setup), indent=2), encoding='utf-8')
    return path
