"""Screenshot-only player: callers execute and acknowledge canvas-coordinate actions.

No desktop capture or input runs here. A caller must supply a fresh BGR canvas
after completing each emitted transaction, with its id as ``acknowledged_id``.
Unacknowledged transactions are never replayed or advanced automatically.

Pass an existing ``Broker`` to keep System One's asynchronous decisions and rule fallbacks.
The caller owns that broker's lifecycle (including ``close``) and generation.
"""

from dataclasses import asdict
import time

import numpy as np

from . import playbook
from .army import Army
from .client import ClientProfile
from .executor import Trust
from .intel import Intel
from .intent import StandingIntent
from .perception import view as views
from .perception.reader import Perception
from .reflex import Reflex
from .repair import Repair
from .skills import Context
from .tasks import TaskRunner
from .world import World


class _FrameTasks(TaskRunner):
    def tick(self, snapshot, view, budget_s=0.03, acknowledged_queue=None):
        deadline = self.clock() + budget_s
        self._check_background(snapshot)
        # Fast units may finish before another army action returns. Let the queue
        # inspect its own acknowledgment frame before another task consumes input.
        for task in sorted(self.tasks.values(), key=lambda t: (t is not acknowledged_queue, t.priority, t.started)):
            if self.executor.pending is not None or self.clock() > deadline:
                break
            self._advance(task, snapshot, view)


class FramePlayer:
    def __init__(self, profile=None, layout=None, clock=time.monotonic, log=None, broker=None, compressed=False,
                 book=None):
        self.clock, self.broker = clock, broker
        self.profile = profile or ClientProfile.load()
        self.profile.use_colors()
        self.layout = layout or self.profile.layout()
        self.perception = Perception(profile=self.profile, layout=self.layout, compressed=compressed)
        self.book = book or playbook.load()
        self.world = World(knowledge=playbook.knowledge(self.book), clock=clock, log=log)
        self.intent = StandingIntent()
        self.pending = None
        self.next_id = 1
        self.runner = _FrameTasks(self, Trust(), clock=clock, log=log)
        self.ctx = Context(self.layout, self.world, self.perception.cursor, pointer=None,
                           build_off_ally=self.profile.raw.get('team_play', {}).get('build_off_ally', False))
        self.ctx.compressed_frames = compressed
        self.intel = Intel(clock, book=self.book)
        self.army = Army(self.ctx, self.runner, self.intent, clock=clock, log=log,
                         intel=self.intel, book=self.book)
        self.reflex = Reflex(self.ctx, self.runner, broker=broker, intent=self.intent, clock=clock, log=log,
                             generation=broker.generation if broker is not None else lambda: 0,
                             army=self.army, book=self.book)
        self.runner.on_failure, self.runner.on_done = self.reflex.failed, self.reflex.done
        self.repair = Repair(self.ctx, self.runner, self.book, clock)

    def act(self, name, lane, steps, commands=1):
        """Capture the one transaction yielded by TaskRunner; never send input."""
        if self.pending is not None:
            raise RuntimeError('An action is still awaiting acknowledgement.')
        serialized = []
        for step in steps:
            data = asdict(step)
            if step.kind == 'repair_mode':
                if not name.startswith('repair:') or step.keys != ('k',):
                    raise ValueError('Repair mode requires a verified repair task.')
                data['kind'] = 'key'
            serialized.append(data)
        self.pending = dict(id=self.next_id, task=name, lane=lane, commands=commands,
                            steps=serialized)
        self.next_id += 1

    def fail(self, transaction_id, reason):
        """Abandon possibly partial input without retrying it on stale assumptions."""
        if self.pending is None or self.pending['id'] != transaction_id:
            raise ValueError('Failure does not match the pending transaction.')
        self.runner.cancel_all(reason)
        self.pending = None

    def step(self, frame, acknowledged_id=None):
        """Observe a canvas and emit at most one new transaction (plain JSON data).

        ``transaction=None, awaiting=id`` means the caller has not acknowledged
        the previous batch. Never execute that batch again. Non-HUD frames cancel
        every task and discard its pending input, even when an ack accompanies it.
        """
        width, height = self.layout.canvas
        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            raise ValueError(f'Expected a {width}x{height} uint8 BGR canvas.')
        if acknowledged_id is not None and (self.pending is None or self.pending['id'] != acknowledged_id):
            raise ValueError('Acknowledgement does not match the pending transaction.')
        self.ctx.last_canvas = frame
        mode, _, _ = self.perception.read(frame, self.world, self.clock())
        if mode['mode'] != 'hud':
            self.runner.cancel_all('game HUD is not visible')
            self.pending = None
            return self._result(mode['mode'])
        if self.pending is not None and acknowledged_id is None:
            return self._result('awaiting_ack')
        acknowledged_queue = None
        if self.pending is not None and self.pending['task'].startswith('queue:'):
            acknowledged_queue = self.runner.tasks.get(self.pending['lane'])
        self.pending = None
        radar = self.world.radar.value
        if radar is not None and radar.online and self.ctx.radar_home is None and radar.own:
            self.ctx.radar_home = radar.own[0][1]
        snapshot = self.world.snapshot()
        previous = views.use(self.layout)
        try:
            if not self.repair.step(snapshot):
                self.reflex.step(snapshot)
                self.army.step(snapshot)
            self.runner.tick(snapshot, None, acknowledged_queue=acknowledged_queue)
            if self.pending is None and not self.runner.tasks and not self.repair.cleanup_pending:
                # A slow transport can finish a census after the next repair scan
                # is due. Give production a turn before that scan takes input again.
                snapshot = self.world.snapshot()
                self.reflex.step(snapshot)
                self.runner.tick(snapshot, None)
        finally:
            views.use(previous)
        return self._result('hud', self.pending)

    def _result(self, state, transaction=None):
        return dict(state=state, transaction=transaction,
                    awaiting=self.pending['id'] if self.pending is not None else None,
                    world=self.world.snapshot(), tasks=self.runner.view(), intent=self.intent.view(),
                    policy='systemone' if self.broker is not None else 'rule')
