"""Is the match actually being played? Only then may the player send input.

Playing requires all of:
- the in-match HUD (static sidebar anchors match),
- no in-game menu (Esc shows red Options/Resume buttons in the cameo area),
- the game holding the pointer (browser pointer lock clips the cursor to the window).
Esc both opens the menu and releases pointer lock, so a paused game fails twice.

The anchors are sidebar crops; where they are on the canvas comes from the layout (the
measured 1024x768 places are the defaults).
"""

from pathlib import Path

import numpy as np

ANCHORS = {'top_buttons': (874, 20, 132, 24), 'sidebar_bottom': (850, 736, 174, 32)}
MENU_REGION = (877, 228, 128, 200)


def _crop(image, box):
    x, y, w, h = box
    return image[y:y + h, x:x + w].astype(np.int16)


class ModeReader:
    def __init__(self, templates, tolerance=12, required=0.7, anchors=None, menu=None, compressed=False):
        self.templates = {k: np.asarray(v, np.int16) for k, v in templates.items()}
        self.tolerance, self.required = tolerance, required
        self.anchors = {k: tuple(v) for k, v in (anchors or ANCHORS).items()}
        self.menu = tuple(menu or MENU_REGION)
        self.compressed = compressed

    @classmethod
    def from_frame(cls, canvas, anchors=None, **kwargs):
        anchors = anchors or ANCHORS
        return cls({name: _crop(canvas, box) for name, box in anchors.items()}, anchors=anchors, **kwargs)

    @classmethod
    def load(cls, path, **kwargs):
        with np.load(Path(path)) as data:
            return cls({k: data[k] for k in data.files}, **kwargs)

    def save(self, path):
        np.savez_compressed(path, **{k: v.astype(np.uint8) for k, v in self.templates.items()})

    def read(self, canvas):
        matches = {}
        for name, template in self.templates.items():
            if name not in self.anchors:
                continue
            compared = np.abs(_crop(canvas, self.anchors[name]) - template).max(axis=-1) <= self.tolerance
            if compared.ndim == 3:
                compared = compared[np.argmax(compared.mean(axis=(1, 2)))]
            matches[name] = compared
        scores = {name: float(match.mean()) for name, match in matches.items()}
        top_score = scores.get('top_buttons', 0)
        if self.compressed and 'top_buttons' in matches:
            # Radar overlays can cross the bottom or left button; require an intact half.
            top = matches['top_buttons']
            scores['top_buttons_upper'] = float(top[:max(1, top.shape[0] // 2)].mean())
            scores['top_buttons_right'] = float(top[:, top.shape[1] // 2:].mean())
            top_score = max(top_score, scores['top_buttons_upper'], scores['top_buttons_right'])
        grid = _crop(canvas, self.menu)
        red = float(np.mean((grid[..., 2] > 150) & (grid[..., 1] < 70) & (grid[..., 0] < 70)))
        bottom_score = scores.get('sidebar_bottom', 0)
        if self.compressed and 'sidebar_bottom' in matches:
            # Building tooltips can cover the left footer while its right half stays intact.
            bottom = matches['sidebar_bottom']
            scores['sidebar_bottom_right'] = float(bottom[:, bottom.shape[1] // 2:].mean())
            bottom_score = max(bottom_score, scores['sidebar_bottom_right'])
        hud = bottom_score >= self.required
        menu = hud and (red > 0.05 or top_score < self.required)
        mode = 'menu' if menu else ('hud' if hud else 'other')
        return dict(mode=mode, scores={k: round(v, 3) for k, v in scores.items()}, menu_red=round(red, 3))


def pointer_locked(pointer, geometry):
    """Browser pointer lock clips the OS cursor to (roughly) the game window."""
    if not pointer or not pointer.get('clipped'):
        return False
    left, top, right, bottom = pointer['clip']
    return (geometry.x - 16 <= left and geometry.y - 16 <= top and
            right <= geometry.x + geometry.width + 16 and bottom <= geometry.y + geometry.height + 16)


def playing(mode, pointer, geometry, foreground):
    """(ok, reason). Input is allowed only while the match is being played."""
    if not foreground:
        return False, 'game window is not in front'
    if mode['mode'] == 'menu':
        return False, 'game menu is open (paused)'
    if mode['mode'] != 'hud':
        return False, 'not in a match (menu, loading, or score screen)'
    if not pointer_locked(pointer, geometry):
        return False, 'game does not hold the pointer (press Resume / click the game)'
    return True, 'playing'
