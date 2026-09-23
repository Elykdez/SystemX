"""Battlefield picture: where the enemy base is, how strong it looked, and what our scouts and
camera looks found, with the time of each observation.

The radar gives positions (its blue dots stay once revealed); camera looks give detail. A
look moves the camera to a radar point for a moment (see army.look) and surveys the view:
enemy objects (the opponent's house color), our own objects (red), and how much of the view
is still shroud. System One reads the picture in words (Reflex.intel_facts): what to look at next,
and how big an attack the enemy's defenses call for; attack targeting falls back to the
remembered enemy base. When a look counts as recent and how long a scout may take are the
System One profile's (playbook section intel).
"""

import numpy as np

from . import playbook
from .perception.radar import distance
from .perception.units import enemy_blobs, house_blobs

HOME_RADIUS = 14                  # radar pixels around home that count as "our base"
BASE_DOTS = 12                    # an enemy dot group this large holds buildings (units are 1-3 dots)
STRENGTH = ((300, 'weak'), (1200, 'medium'), (float('inf'), 'strong'))   # enemy pixels in one view


def survey(canvas, tactical, compressed=False):
    """What one camera view shows: enemy and own objects, and the shrouded share."""
    region = (tactical.x, tactical.y, tactical.w, tactical.h)
    enemy = [b for b in enemy_blobs(canvas, region, compressed=compressed) if b[0] >= 12]
    own = [b for b in house_blobs(canvas, region) if b[0] >= 20]
    view = canvas[tactical.y:tactical.y + tactical.h, tactical.x:tactical.x + tactical.w]
    shroud = float(np.mean(view.astype(np.int16).sum(axis=2) < 24))
    return dict(enemy_objects=len(enemy), enemy_pixels=int(sum(b[0] for b in enemy)),
                own_objects=len(own), shroud=round(shroud, 2))


def strength_of(pixels):
    return next(name for limit, name in STRENGTH if pixels < limit)


class Intel:
    def __init__(self, clock, book=None):
        self.clock = clock
        self.plan = (book or playbook.default())['intel']
        self.enemy_base = None        # radar point of the main enemy group (canvas pixels)
        self.enemy_area = 0           # its radar dot area when last seen
        self.found_at = None
        self.looks = {}               # 'enemy_base' | 'army' -> dict(at, point, survey...)
        self.scout = None             # dict(at, target) while a scout is out
        self.strength = None          # 'weak' | 'medium' | 'strong', from the last look at the enemy base

    # -- updates ----------------------------------------------------------------
    def update(self, radar, home, now=None):
        """The enemy base is the largest enemy group on the radar away from our base, and large
        enough to be buildings (live game 12: a roaming unit remembered as "the base" kept every
        wave going back to an empty spot mid-map)."""
        now = now if now is not None else self.clock()
        if radar is None or not radar.online:
            return
        far = [e for e in radar.enemy if e[0] >= BASE_DOTS and
               (home is None or distance(e[1], home) > 2 * HOME_RADIUS)]
        if not far:
            return
        area, point = max(far)
        if self.enemy_base is None:
            self.found_at = now
            self.enemy_base, self.enemy_area = point, area
        elif distance(point, self.enemy_base) <= 10 or area >= 0.7 * self.enemy_area:
            self.enemy_base, self.enemy_area = point, area    # same base, or a bigger group elsewhere
        if self.scout is not None:
            self.scout['found'] = True

    def record_look(self, kind, point, seen, now=None):
        now = now if now is not None else self.clock()
        self.looks[kind] = dict(at=now, point=list(point), **seen)
        if kind == 'enemy_base' and seen['shroud'] < 0.6:
            self.strength = strength_of(seen['enemy_pixels'])

    def send_scout(self, target, now=None):
        self.scout = dict(at=now if now is not None else self.clock(), target=list(target), found=False)

    # -- questions ------------------------------------------------------------------
    def looked(self, kind, now):
        """Seconds since the last look of this kind, or None."""
        look = self.looks.get(kind)
        return None if look is None else now - look['at']

    def stale(self, kind, now):
        age = self.looked(kind, now)
        return age is None or age >= self.plan['look_again'][kind]

    def scout_out(self, now):
        return (self.scout is not None and not self.scout.get('found') and
                now - self.scout['at'] < self.plan['scout_patience'])

    def summary(self, now=None):
        now = now if now is not None else self.clock()
        return dict(enemy_base=self.enemy_base, strength=self.strength,
                    found=None if self.found_at is None else round(now - self.found_at, 1),
                    scout=self.scout, looks={k: dict(v, age=round(now - v['at'], 1)) for k, v in self.looks.items()})
