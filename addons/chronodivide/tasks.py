"""Skill coroutines: interleaved per-lane tasks with perception between input steps.

A skill is a generator that yields instructions:
  Act(steps)            send one guarded transaction (then yield control to other lanes)
  Wait(cond, timeout)   resume when cond(snapshot, view) is true; fail on timeout
  Poll(cond, timeout)   like Wait, but resumes with False on timeout instead of failing
  Look(fn)              resume with fn(snapshot, view) evaluated on the current tick
  Expect(check, timeout, name)
                        strict skills wait for check(snapshot) -> True; proven skills
                        continue at once while the check runs in the background
Only one task runs per lane, so production, construction, and army work interleave
while dependent steps inside a task stay ordered.
"""

from dataclasses import dataclass, field
import time

from desktop.target import DesktopPause


@dataclass
class Act:
    steps: list
    commands: int = 1


@dataclass
class Wait:
    until: object
    timeout: float
    reason: str = ''


@dataclass
class Poll:
    until: object
    timeout: float
    reason: str = ''


@dataclass
class Look:
    fn: object


@dataclass
class Expect:
    check: object
    timeout: float
    name: str


class TaskFailed(RuntimeError):
    pass


@dataclass
class Task:
    name: str
    lane: str
    generator: object
    priority: int = 5
    started: float = field(default_factory=time.monotonic)
    status: str = 'running'          # running | waiting | done | failed | cancelled
    waiting: object = None
    wait_since: float = 0.0
    reason: str = ''
    send_value: object = None
    acts: int = 0


@dataclass
class Background:
    task: str
    lane: str
    check: object
    deadline: float
    started: float


class TaskRunner:
    def __init__(self, executor, trust, clock=time.monotonic, log=None, on_failure=None, on_done=None):
        self.executor, self.trust, self.clock, self.log = executor, trust, clock, log
        self.on_failure, self.on_done = on_failure, on_done
        self.tasks = {}                  # lane -> Task
        self.background = []
        self.finished = []               # bounded history for the controller

    def busy(self, lane):
        return lane in self.tasks

    def start(self, name, lane, generator, priority=5, preempt=False):
        current = self.tasks.get(lane)
        if current is not None:
            if not preempt or current.priority <= priority:
                return False
            self._end(current, 'cancelled', f'preempted by {name}')
        self.tasks[lane] = Task(name, lane, generator, priority, self.clock())
        return True

    def cancel_all(self, reason):
        for task in list(self.tasks.values()):
            self._end(task, 'cancelled', reason)
        self.background.clear()

    def _end(self, task, status, reason=''):
        task.status, task.reason = status, reason
        self.tasks.pop(task.lane, None)
        try:
            task.generator.close()
        except Exception:
            pass
        self.finished.append(task)
        del self.finished[:-50]
        if self.log:
            self.log(dict(time=time.time(), task=task.name, lane=task.lane, status=status, reason=reason,
                          acts=task.acts, ms=round((self.clock() - task.started) * 1000, 1)))
        if status == 'failed' and self.on_failure:
            self.on_failure(task)
        if status == 'done' and self.on_done:
            self.on_done(task)

    def _check_background(self, snapshot):
        now, remaining = self.clock(), []
        for item in self.background:
            outcome = item.check(snapshot)
            if outcome is True:
                self.trust.update(item.task, True)
            elif outcome is False or now >= item.deadline:
                self.trust.update(item.task, False)
                if self.log:
                    self.log(dict(time=time.time(), expectation=item.task, lane=item.lane,
                                  outcome='violated' if outcome is False else 'timeout'))
                if self.on_failure:
                    self.on_failure(Task(item.task, item.lane, None, status='failed',
                                         reason='background expectation failed'))
            else:
                remaining.append(item)
        self.background = remaining

    def tick(self, snapshot, view, budget_s=0.03):
        """Advance tasks in priority order; each gets at most one Act per tick."""
        deadline = self.clock() + budget_s
        self._check_background(snapshot)
        for task in sorted(self.tasks.values(), key=lambda t: (t.priority, t.started)):
            if self.clock() > deadline:
                break
            self._advance(task, snapshot, view)

    def _advance(self, task, snapshot, view):
        now = self.clock()
        waiting = task.waiting
        if isinstance(waiting, Poll):
            if waiting.until(snapshot, view):
                task.waiting, task.send_value = None, True
            elif now - task.wait_since > waiting.timeout:
                task.waiting, task.send_value = None, False
            else:
                return
        elif isinstance(waiting, Wait):
            if waiting.until(snapshot, view):
                task.waiting, task.send_value = None, True
            elif now - task.wait_since > waiting.timeout:
                return self._end(task, 'failed', f'timeout waiting: {waiting.reason}')
            else:
                return
        elif isinstance(waiting, Expect):
            outcome = waiting.check(snapshot)
            if outcome is True:
                self.trust.update(task.name, True)
                task.waiting, task.send_value = None, True
            elif outcome is False or now - task.wait_since > waiting.timeout:
                self.trust.update(task.name, False)
                return self._end(task, 'failed', f'expectation {waiting.name} '
                                 + ('violated' if outcome is False else 'timed out'))
            else:
                return
        while True:
            try:
                instruction = task.generator.send(task.send_value)
            except StopIteration:
                return self._end(task, 'done')
            except TaskFailed as error:
                return self._end(task, 'failed', str(error))
            except DesktopPause:
                self._end(task, 'cancelled', 'paused')
                raise
            task.send_value = None
            if isinstance(instruction, Look):
                task.send_value = instruction.fn(snapshot, view)
                continue
            if isinstance(instruction, Act):
                try:
                    self.executor.act(task.name, task.lane, instruction.steps, instruction.commands)
                except TaskFailed as error:
                    return self._end(task, 'failed', str(error))
                except DesktopPause:
                    self._end(task, 'cancelled', 'paused before or during input')
                    raise
                task.acts += 1
                return
            if isinstance(instruction, Poll):
                if instruction.until(snapshot, view):
                    task.send_value = True
                    continue
                task.waiting, task.wait_since = instruction, now
                return
            if isinstance(instruction, Wait):
                if instruction.until(snapshot, view):
                    task.send_value = True
                    continue
                task.waiting, task.wait_since = instruction, now
                return
            if isinstance(instruction, Expect):
                if self.trust.level(task.name) == 'optimistic':
                    self.background.append(Background(task.name, task.lane, instruction.check,
                                                      now + instruction.timeout, now))
                    task.send_value = True
                    continue
                if instruction.check(snapshot) is True:   # effect already visible
                    self.trust.update(task.name, True)
                    task.send_value = True
                    continue
                task.waiting, task.wait_since = instruction, now
                return
            raise TypeError(f'Unknown task instruction {instruction!r}')

    def view(self):
        return dict(active={lane: dict(task=t.name, waiting=type(t.waiting).__name__ if t.waiting else None,
                                       reason=getattr(t.waiting, 'reason', '') or getattr(t.waiting, 'name', ''),
                                       acts=t.acts) for lane, t in self.tasks.items()},
                    background=len(self.background),
                    recent=[dict(task=t.name, lane=t.lane, status=t.status, reason=t.reason)
                            for t in self.finished[-8:]])
