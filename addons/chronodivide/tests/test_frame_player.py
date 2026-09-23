import json

import numpy as np
import pytest

from addons.chronodivide.executor import Step
from addons.chronodivide.frame_player import FramePlayer
from addons.chronodivide.skills import queue_item
from addons.chronodivide.tasks import Act, Look
from addons.chronodivide.world import Slot


@pytest.fixture
def player():
    player = FramePlayer(clock=lambda: 10.0)

    def read(frame, world, now):
        mode = 'hud' if frame[0, 0, 0] != 2 else 'menu'
        world.set('mode', mode, now)
        world.set('sidebar_empty', frame[0, 0, 0] == 0, now)
        if frame[0, 0, 0] == 1:
            world.update_sidebar('buildings', [Slot(0, 'tesla_reactor', 'available')], now)
        return dict(mode=mode), [], None

    player.perception.read = read
    return player


def frame(marker=0):
    pixels = np.zeros((768, 1024, 3), np.uint8)
    pixels[0, 0, 0] = marker
    return pixels


def test_deploy_waits_for_ack_and_uses_new_frame(player):
    first = player.step(frame())
    assert first['transaction']['task'] == 'deploy_mcv'
    steps = first['transaction']['steps']
    assert [s['keys'] for s in steps if s['kind'] == 'key'] == [('n',), ('s',), ('d',)]
    assert [s['ms'] for s in steps if s['kind'] == 'wait'] == [150, 150]
    assert json.loads(json.dumps(first))['awaiting'] == 1
    # Even new visual evidence must not advance a skill before input is acknowledged.
    waiting = player.step(frame(1))
    assert waiting['transaction'] is None and waiting['awaiting'] == 1
    assert not player.world.mcv_deployed.value
    second = player.step(frame(1), acknowledged_id=1)
    assert player.world.mcv_deployed.value
    assert second['transaction']['steps'][0]['keys'] == ('h',)
    assert second['transaction']['id'] == 2


def test_single_batch_across_lanes_and_fresh_look(player):
    player.reflex.step = lambda snapshot: None
    player.army.step = lambda snapshot: None
    seen = []

    def task():
        yield Act([Step.key('p')])
        seen.append((yield Look(lambda s, v: int(player.ctx.last_canvas[0, 0, 0]))))
        yield Act([Step.key('s')])

    player.runner.start('one', 'army', task(), priority=1)
    player.runner.start('two', 'sidebar', task(), priority=2)
    first = player.step(frame())
    assert player.runner.tasks['sidebar'].acts == 0
    assert player.step(frame(1))['transaction'] is None
    assert not seen
    second = player.step(frame(1), acknowledged_id=first['awaiting'])
    assert seen == [1] and second['transaction']['task'] == 'one'
    assert player.runner.tasks['sidebar'].acts == 0


def test_menu_discards_unacknowledged_input(player):
    player.step(frame())
    result = player.step(frame(2))
    assert result['state'] == 'menu' and result['awaiting'] is None
    assert not player.runner.tasks and not player.runner.background
    with pytest.raises(ValueError, match='Acknowledgement'):
        player.step(frame(), acknowledged_id=1)


@pytest.mark.parametrize('cleanup_pending', [False, True])
def test_finished_army_task_releases_input_to_production_in_the_same_frame(player, cleanup_pending):
    player.world.set('credits', 5000)
    player.world.set('mcv_deployed', True)
    player.world.structures['construction_yard'] = 1
    player.army.step = lambda snapshot: None
    player.repair.cleanup_pending = cleanup_pending
    player.runner.start('army:census', 'army', (step for step in ()), priority=2)
    result = player.step(frame(1))
    if cleanup_pending:
        assert result['transaction'] is None
    else:
        assert result['transaction']['task'] == 'queue:tesla_reactor'
        waiting = player.step(frame(1))
        assert waiting['state'] == 'awaiting_ack' and waiting['transaction'] is None


@pytest.mark.parametrize('building_on_ack', [True, False])
def test_fast_unit_observed_before_competing_army_action(player, building_on_ack):
    player.reflex.step = lambda snapshot: None
    player.army.step = lambda snapshot: None

    def read(pixels, world, now):
        world.set('mode', 'hud', now)
        state = 'building' if pixels[0, 0, 0] == 1 else 'available'
        world.update_sidebar('infantry', [Slot(0, 'conscript', state)], now)
        return dict(mode='hud'), [], None

    def census():
        yield Act([Step.key('p')])

    player.perception.read = read
    player.runner.start('queue:conscript', 'sidebar', queue_item(player.ctx, 'conscript'), priority=3)
    queued = player.step(frame())
    player.runner.start('army:census', 'army', census(), priority=2)
    # An unacknowledged build observation cannot confirm input. Only the ack's frame counts.
    assert player.step(frame(1))['transaction'] is None
    assert player.world.units_built == {}
    acknowledged = player.step(frame(int(building_on_ack)), acknowledged_id=queued['awaiting'])
    assert acknowledged['transaction']['task'] == 'army:census'
    assert player.world.units_built.get('conscript', 0) == int(building_on_ack)
    assert player.runner.busy('sidebar') is not building_on_ack
    # The fast unit has finished by the next observation; a confirmed queue stays confirmed.
    player.step(frame(), acknowledged_id=acknowledged['awaiting'])
    assert player.world.units_built.get('conscript', 0) == int(building_on_ack)


def test_army_ack_keeps_other_task_priority(player):
    player.reflex.step = lambda snapshot: None
    player.army.step = lambda snapshot: None

    def army():
        yield Act([Step.key('p')])
        yield Act([Step.key('s')])

    def sidebar():
        yield Act([Step.key('e')])

    player.runner.start('army:census', 'army', army(), priority=3)
    selected = player.step(frame())
    player.runner.start('look:infantry', 'sidebar', sidebar(), priority=2)
    next_action = player.step(frame(), acknowledged_id=selected['awaiting'])
    assert next_action['transaction']['task'] == 'look:infantry'


def test_failed_transport_cancels_without_replaying(player):
    player.step(frame())
    with pytest.raises(ValueError, match='Failure'):
        player.fail(99, 'failed input')
    player.fail(1, 'failed input')
    assert player.pending is None and not player.runner.tasks
    assert player.runner.finished[-1].reason == 'failed input'


def test_wrong_ack_does_not_advance(player):
    player.step(frame())
    with pytest.raises(ValueError, match='Acknowledgement'):
        player.step(frame(1), acknowledged_id=99)
    assert player.pending['id'] == 1 and not player.world.mcv_deployed.value


def test_rejects_uncropped_screenshot(player):
    with pytest.raises(ValueError, match='1024x768'):
        player.step(np.zeros((900, 1100, 3), np.uint8))


def test_compressed_frames_are_explicitly_opted_in():
    strict, compressed = FramePlayer(), FramePlayer(compressed=True)
    assert not strict.ctx.compressed_frames and compressed.ctx.compressed_frames
    assert not strict.perception.compressed and compressed.perception.compressed
    assert not strict.perception.catalog.compressed and compressed.perception.catalog.compressed
    assert not strict.perception.credits.compressed and compressed.perception.credits.compressed


@pytest.mark.parametrize('unavailable', [False, True])
def test_async_systemone_or_rule_fallback_cannot_acknowledge_pending_input(player, unavailable):
    import threading
    from addons.chronodivide.broker import Broker

    release, entered, resolved = threading.Event(), threading.Event(), threading.Event()

    def transport(endpoint, body):
        entered.set()
        assert release.wait(2)
        if unavailable:
            raise ConnectionError('System One unavailable')
        return dict(answers={qid: dict(type='choice', choice='medium' if qid == 'attack_size'
                                      else next(iter(question['criteria'])))
                             for qid, question in body['questions'].items()})

    broker = Broker(['test-server'], transport=transport, generation=lambda: 7)
    try:
        connected = FramePlayer(broker=broker)
        connected.perception.read = player.perception.read
        callback = connected.reflex._strategy_result

        def received(decision, answers, provenance, meta):
            callback(decision, answers, provenance, meta)
            if decision.key == 'strategy:attack_size':
                resolved.set()

        connected.reflex._strategy_result = received
        first = connected.step(frame())
        assert first['transaction']['task'] == 'deploy_mcv' and first['policy'] == 'systemone'
        assert entered.wait(2)               # transport is waiting while the action has already been emitted
        pending = json.dumps(first['transaction'])
        release.set()
        assert resolved.wait(2)
        waiting = connected.step(frame(1))
        assert waiting['transaction'] is None and waiting['awaiting'] == first['awaiting']
        assert json.dumps(connected.pending) == pending
        assert not connected.world.mcv_deployed.value
        expected_size = connected.book.option('attack_size', 'large' if unavailable else 'medium')['tanks']
        assert waiting['intent']['attack_size'] == expected_size
        assert waiting['intent']['provenance']['attack_size'] == ('default-after-error' if unavailable else 'systemone')
        acknowledged = connected.step(frame(1), acknowledged_id=first['awaiting'])
        assert connected.world.mcv_deployed.value
        assert acknowledged['transaction']['id'] == first['awaiting'] + 1
    finally:
        release.set()
        broker.close()
