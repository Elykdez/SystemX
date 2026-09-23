"""Unattended execute mode: the agent arms and resumes itself.

For runs where nobody presses Act: the agent arms when the match is being played.
Physical mouse or keyboard use still pauses it at once. After `resume_after` quiet
seconds it re-arms, and the pointer is calibrated again, because someone moving the
mouse under pointer lock also moves the game cursor.

Pointer capture (read from the game's code): the game requests pointer lock only from a
click on its canvas; the first such click grants permission, after which resuming the
match re-locks, and a canvas click re-locks after a loss. So when the HUD is up but the
pointer is free, the supervisor clicks the map area (after a right click that clears any
selection), backing off between attempts. The game's pause menu is resumed with Resume
Mission, unless the user opened it (physical input just before it): that one is left
alone until `user_menu_after` quiet seconds. F10 (or Pause) holds the agent paused until
F8 (or Act), which also resumes a user's menu.
"""

import time


class Supervisor:
    def __init__(self, player, resume_after=8.0, grab_after=2.0, menu_after=1.5, user_menu_after=60.0,
                 clock=time.monotonic, log=None, max_attempts=5, internal_after=3.0, front=None):
        self.player, self.resume_after, self.grab_after = player, resume_after, grab_after
        self.menu_after, self.user_menu_after = menu_after, user_menu_after
        self.clock, self.log, self.max_attempts = clock, log, max_attempts
        self.internal_after = internal_after
        self.front = front
        self.front_at = -1e9
        self.held = False                # user pause (F10): no automatic resume
        self.forced = False              # F8 after a pause: resume even a menu the user opened
        self.free_since = None           # when the pointer was first seen free (or last recapture try)
        self.attempts = 0                # recapture clicks since the pointer was last held
        self.counts = dict(resumes=0, recaptures=0, menu_resumes=0)

    def hold(self, reason='paused by the user'):
        self.held = True
        self.player.pause(reason)

    def release(self):
        self.held, self.forced, self.attempts = False, True, 0
        self.player.arm_on_start()

    def quiet(self, now, seconds=None):
        last = self.player.last_physical()
        return last is None or now - last >= (self.resume_after if seconds is None else seconds)

    def step(self):
        """Called a few times per second. Returns what it did, or None."""
        if self.held:
            self.free_since = None
            return None
        now = self.clock()
        player = self.player
        if self.front is not None and not player.in_front():
            self.free_since = None
            if self.quiet(now) and now - self.front_at >= self.resume_after:
                self.front_at = now
                if self.front():
                    return self._note('restored-window')
            return None
        if player.taken_over():
            self.free_since = None
            if not self.quiet(now):
                return None
            player.arm_on_start()        # arms now if playing, else as soon as play resumes
            self.counts['resumes'] += 1
            return self._note('resumed')
        since = getattr(player, 'paused_internally', lambda: None)()
        if since is not None and not player.menu_open():
            # A pause the agent caused itself: nobody would ever press F8 for it (live game 28 idled
            # fifteen minutes after one). Resume after a moment; the pointer is re-found first.
            if now - since < self.internal_after or not self.quiet(now):
                return None
            player.arm_on_start()
            self.counts['resumes'] += 1
            return self._note('resumed-internal')
        if player.menu_open():
            self.free_since = None
            if self.forced:                  # F8 / Act: the user asked to go on
                ready = True
            elif player.menu_by_user:
                ready = now - player.menu_since >= self.user_menu_after and self.quiet(now, self.user_menu_after)
            else:
                ready = now - player.menu_since >= self.menu_after and self.quiet(now)
            if not ready:
                return None
            player.resume_menu()
            self.forced = False
            self.counts['menu_resumes'] += 1
            return self._note('resumed-menu')
        if not player.wants_pointer():
            self.free_since = None
            if player.last_playing[0]:
                self.attempts = 0            # the game holds the pointer again
            return None
        if self.free_since is None:
            self.free_since = now
        # First try at once (match start: the MCV waits for it), then back off: 4 s, 6 s, ...
        wait = 0.5 if self.attempts == 0 else self.grab_after * (1 + self.attempts)
        if now - self.free_since < wait or not self.quiet(now) or self.attempts >= self.max_attempts:
            return None
        self.free_since = now
        player.recapture_pointer()
        self.attempts += 1
        self.counts['recaptures'] += 1
        return self._note('recaptured')

    def _note(self, what):
        if self.log:
            self.log(dict(time=time.time(), supervisor=what, **self.counts))
        return what
