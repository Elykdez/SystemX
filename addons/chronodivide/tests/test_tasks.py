import pytest

from desktop.target import DesktopPause
from addons.chronodivide.executor import Step, Trust
from addons.chronodivide.tasks import Act, Expect, Look, TaskFailed, TaskRunner, Wait


class FakeExecutor:
    def __init__(self, fail=False):
        self.acts, self.fail = [], fail

    def act(self, name, lane, steps, commands=1):
        if self.fail:
            raise DesktopPause('focus lost')
        self.acts.append((name, lane, len(steps)))


def runner(trust=None, clock=None):
    executor = FakeExecutor()
    failures = []
    time = clock or [0.0]
    tasks = TaskRunner(executor, trust or Trust(), clock=lambda: time[0], on_failure=failures.append)
    return tasks, executor, failures, time


def test_lanes_interleave_one_act_per_tick():
    tasks, executor, _, _ = runner()

    def two_acts(tag):
        yield Act([Step.key('d')])
        yield Act([Step.key('s')])
    tasks.start('a', 'vehicles', two_acts('a'))
    tasks.start('b', 'infantry', two_acts('b'))
    tasks.tick({}, None)
    assert [a[1] for a in executor.acts] == ['vehicles', 'infantry']
    tasks.tick({}, None)
    tasks.tick({}, None)
    assert len(executor.acts) == 4 and not tasks.tasks


def test_one_task_per_lane_and_priority_preemption():
    tasks, *_ = runner()

    def idle():
        yield Wait(lambda s, v: False, 10)
    assert tasks.start('low', 'buildings', idle(), priority=5)
    assert not tasks.start('same', 'buildings', idle(), priority=5)
    assert tasks.start('urgent', 'buildings', idle(), priority=1, preempt=True)
    assert tasks.tasks['buildings'].name == 'urgent'


def test_wait_and_look_feed_values_back_into_the_skill():
    tasks, executor, _, time = runner()
    seen = []

    def skill():
        value = yield Look(lambda snapshot, view: snapshot['x'] * 2)
        seen.append(value)
        yield Wait(lambda snapshot, view: snapshot['ready'], 1.0, 'ready')
        yield Act([Step.key('d')])
    tasks.start('s', 'misc', skill())
    tasks.tick({'x': 21, 'ready': False}, None)
    assert seen == [42] and not executor.acts
    tasks.tick({'x': 0, 'ready': True}, None)
    assert executor.acts


def test_strict_expectation_blocks_then_times_out_and_reports_failure():
    tasks, executor, failures, time = runner()

    def skill():
        yield Act([Step.key('d')])
        yield Expect(lambda s: s.get('ok'), 0.5, 'deployed')
        yield Act([Step.key('s')])
    tasks.start('deploy', 'misc', skill())
    tasks.tick({}, None)
    tasks.tick({}, None)
    assert len(executor.acts) == 1
    time[0] = 1.0
    tasks.tick({}, None)
    assert failures and failures[0].reason.startswith('expectation deployed')
    assert tasks.trust.level('deploy') == 'strict'


def test_proven_skill_continues_and_checks_in_background():
    trust = Trust(promote_after=1)
    trust.update('queue', True)
    tasks, executor, failures, time = runner(trust=trust)

    def skill():
        yield Act([Step.key('d')])
        yield Expect(lambda s: s.get('ok'), 0.5, 'queued')
        yield Act([Step.key('s')])
    tasks.start('queue', 'vehicles', skill())
    tasks.tick({}, None)
    tasks.tick({}, None)
    assert len(executor.acts) == 2 and len(tasks.background) == 1
    time[0] = 1.0
    tasks.tick({}, None)
    assert failures and trust.level('queue') == 'strict'


def test_skill_failure_and_pause_end_the_task():
    tasks, executor, failures, _ = runner()

    def broken():
        raise TaskFailed('no slot')
        yield
    tasks.start('b', 'misc', broken())
    tasks.tick({}, None)
    assert failures[0].reason == 'no slot'
    tasks.executor = FakeExecutor(fail=True)

    def acting():
        yield Act([Step.key('d')])
    tasks.start('a', 'misc', acting())
    with pytest.raises(DesktopPause):
        tasks.tick({}, None)
    assert not tasks.tasks
