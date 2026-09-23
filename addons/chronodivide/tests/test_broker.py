import threading
import time

from addons.chronodivide.broker import Broker, Decision


QUESTION = {'q': {'type': 'choice', 'instructions': 'Pick.', 'criteria': {'a': 'A', 'b': 'B'}}}
DEFAULT = {'q': {'type': 'choice', 'choice': 'a'}}


class FakeTransport:
    def __init__(self, delay=0.0, choice='b'):
        self.delay, self.choice = delay, choice
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, endpoint, body):
        with self.lock:
            self.calls.append((endpoint, next(iter(body['questions']))))
        time.sleep(self.delay)
        return {'answers': {q: {'type': 'choice', 'choice': self.choice} for q in body['questions']},
                'latency_ms': self.delay * 1000, 'usage': {'input_tokens': 10}}


def collect():
    results, done = [], threading.Event()

    def callback(decision, answers, provenance, meta):
        results.append((decision.key, answers and answers['q']['choice'], provenance))
        done.set()
    return results, done, callback


def decision(key, callback, kind='production', deadline=1.0, **extra):
    return Decision(key, kind, {'s': 1}, QUESTION, time.monotonic() + deadline, callback, DEFAULT, **extra)


def test_valid_answer_resolves_as_systemone():
    results, done, callback = collect()
    broker = Broker(['e0'], FakeTransport())
    broker.submit(decision('k', callback))
    assert done.wait(2)
    broker.close()
    assert results == [('k', 'b', 'systemone')]


def test_out_of_list_choice_falls_back_to_default():
    results, done, callback = collect()
    broker = Broker(['e0'], FakeTransport(choice='zzz'))
    broker.submit(decision('k', callback))
    assert done.wait(2)
    broker.close()
    assert results == [('k', 'a', 'default-after-error')]


def test_deadline_applies_default_and_late_answer_is_discarded():
    results, done, callback = collect()
    logs = []
    broker = Broker(['e0'], FakeTransport(delay=0.3), log=logs.append)
    broker.submit(decision('k', callback, deadline=0.05))
    assert done.wait(2)
    time.sleep(0.4)
    broker.close()
    assert results == [('k', 'a', 'default-after-deadline')]
    assert any(entry.get('provenance') == 'late' for entry in logs)


def test_newer_decision_replaces_unsent_one():
    results, _, callback = collect()
    transport = FakeTransport(delay=0.1)
    broker = Broker(['e0'], transport)
    broker.submit(decision('busy', callback, kind='urgent'))
    time.sleep(0.02)
    broker.submit(decision('lane', callback))
    broker.submit(decision('lane', callback))
    time.sleep(0.5)
    broker.close()
    assert [r for r in results if r[0] == 'lane'] == [('lane', 'b', 'systemone')]
    assert broker.summary()['production']['superseded'] == 1


def test_first_route_is_reserved_for_urgent_work():
    results, _, callback = collect()
    transport = FakeTransport(delay=0.05)
    broker = Broker(['urgent-route', 'shared-route'], transport)
    for index in range(4):
        broker.submit(decision(f'strategy{index}', callback, kind='strategy'))
    time.sleep(0.6)
    broker.close()
    assert {endpoint for endpoint, _ in transport.calls} == {'shared-route'}
    assert len(results) == 4


def test_cached_answer_is_reused_with_provenance():
    results, _, callback = collect()
    broker = Broker(['e0'], FakeTransport(), memo_seconds=5)
    broker.submit(decision('k', callback, cache_key=('same',)))
    time.sleep(0.2)
    broker.submit(decision('k', callback, cache_key=('same',)))
    time.sleep(0.2)
    broker.close()
    assert [r[2] for r in results] == ['systemone', 'systemone-cached']


def test_stale_generation_is_never_delivered():
    results, _, callback = collect()
    generation = [1]
    broker = Broker(['e0'], FakeTransport(delay=0.1), generation=lambda: generation[0])
    broker.submit(decision('k', callback, generation=1))
    time.sleep(0.02)
    generation[0] = 2
    time.sleep(0.3)
    broker.close()
    assert results == []


def test_priority_order_urgent_before_strategy():
    order = []
    started = threading.Event()

    def callback(decision, answers, provenance, meta):
        order.append(decision.key)
    transport = FakeTransport(delay=0.05)
    broker = Broker(['e0'], transport, reserve_urgent=False)
    broker.submit(decision('first', callback, kind='production'))
    time.sleep(0.01)
    broker.submit(decision('strategy', callback, kind='strategy'))
    broker.submit(decision('urgent', callback, kind='urgent'))
    time.sleep(0.5)
    broker.close()
    assert order.index('urgent') < order.index('strategy')
