"""System One broker: multi-route, deadline-bounded decisions that never block the player.

Each route keeps one request in flight; by default there is one route per server
worker (the core server runs parallel model replicas). Decisions wait in
replaceable per-key slots; urgent kinds can reserve the first route so strategic
packets never delay a threat response. A decision is resolved exactly
once: by a valid System One answer, a cached answer, or its rule default when the
deadline passes or the request fails. Late answers are logged and discarded.
"""

from model_config import MODEL_NAME

from collections import defaultdict, deque
from dataclasses import dataclass, field
import itertools
import threading
import time

from systemone_client import Busy, SystemOneClient

KINDS = ('urgent', 'tactical', 'production', 'strategy')
PRIORITY = {kind: index for index, kind in enumerate(KINDS)}


@dataclass
class Decision:
    key: str
    kind: str
    state: dict
    questions: dict
    deadline: float
    on_result: object
    default: dict | None = None
    generation: int = 0
    cache_key: object = None
    optional: bool = False
    created: float = field(default_factory=time.monotonic)
    sequence: int = 0
    resolved: bool = False
    sent_at: float | None = None

    def __post_init__(self):
        if self.kind not in PRIORITY:
            raise ValueError(f'Unknown decision kind {self.kind}')


def valid_answers(questions, answers):
    """Choice answers must name an offered candidate; anything else is rejected."""
    for qid, question in questions.items():
        answer = answers.get(qid)
        if not isinstance(answer, dict):
            return False
        if question['type'] == 'choice' and answer.get('choice') not in question['criteria']:
            return False
        if question['type'] == 'score' and not isinstance(answer.get('score'), (int, float)):
            return False
        if question['type'] == 'noul' and not isinstance(answer.get('noul'), (int, float)):
            return False
    return True


class HttpTransport:
    """One shared thread-safe SystemX client per endpoint URL."""

    def __init__(self, timeout=10.0, api_key=None):
        self.timeout, self.api_key = timeout, api_key
        self.clients = {}

    def __call__(self, endpoint, body):
        client = self.clients.get(endpoint)
        if client is None:
            client = self.clients[endpoint] = SystemOneClient(endpoint, self.timeout, self.api_key)
        # Splitting takes every free worker, so only urgent work may use it (identical answers).
        return client.decide(body['state'], body['questions'], parallel=body.get('parallel', False))

    def close(self):
        for client in self.clients.values():
            client.close()


def server_routes(server):
    """Routes for one server: one per ready worker (at least one)."""
    with SystemOneClient(server, timeout=3) as client:
        ready = client.workers().get('ready', 1)
    return [server] * max(1, ready)


class Broker:
    def __init__(self, endpoints, transport=None, reserve_urgent=True, memo_seconds=0.0,
                 generation=lambda: 0, clock=time.monotonic, log=None):
        if not endpoints:
            raise ValueError('At least one System One endpoint is required.')
        self.endpoints = list(endpoints)
        self.transport = transport or HttpTransport()
        self.reserve_urgent = reserve_urgent and len(self.endpoints) > 1
        self.memo_seconds = memo_seconds
        self.generation, self.clock, self.log = generation, clock, log
        self.lock = threading.Condition()
        self.pending = {}
        self.inflight = {}
        self.memo = {}
        self.counter = itertools.count()
        self.stats = defaultdict(lambda: defaultdict(int))
        self.latency = defaultdict(lambda: deque(maxlen=200))
        self.busy = defaultdict(float)
        self.started = clock()
        self.closed = False
        self.callbacks = deque()
        self.workers = [threading.Thread(target=self._work, args=(i,), name=f'systemone-route-{i}', daemon=True)
                        for i in range(len(self.endpoints))]
        self.timer = threading.Thread(target=self._deadlines, name='systemone-deadlines', daemon=True)
        self.dispatcher = threading.Thread(target=self._dispatch, name='systemone-callbacks', daemon=True)
        for thread in (*self.workers, self.timer, self.dispatcher):
            thread.start()

    # -- submission -------------------------------------------------------
    def submit(self, decision):
        now = self.clock()
        with self.lock:
            decision.sequence = next(self.counter)
            if decision.cache_key is not None and self.memo_seconds > 0:
                cached = self.memo.get((decision.key, decision.cache_key))
                if cached and now - cached[1] <= self.memo_seconds:
                    self._resolve(decision, cached[0], 'systemone-cached', dict(age_ms=round((now - cached[1]) * 1000, 1)))
                    return
            replaced = self.pending.get(decision.key)
            if replaced is not None:
                replaced.resolved = True
                self.stats[replaced.kind]['superseded'] += 1
            self.pending[decision.key] = decision
            self.lock.notify_all()

    def busy_keys(self):
        with self.lock:
            return set(self.pending) | {d.key for d in self.inflight.values()}

    # -- routing ----------------------------------------------------------
    def _eligible(self, route, decision):
        if self.reserve_urgent and route == 0:
            return decision.kind in ('urgent', 'tactical')
        return True

    def _next(self, route):
        choices = [d for d in self.pending.values() if self._eligible(route, d)]
        if not choices:
            return None
        required = [d for d in choices if not d.optional]
        pool = required or choices
        return min(pool, key=lambda d: (PRIORITY[d.kind], d.deadline, d.sequence))

    def _work(self, route):
        endpoint = self.endpoints[route]
        while True:
            with self.lock:
                decision = None
                while not self.closed:
                    decision = self._next(route)
                    if decision is not None:
                        break
                    self.lock.wait(0.05)
                if self.closed:
                    return
                del self.pending[decision.key]
                if decision.resolved:
                    continue
                if decision.generation != self.generation():
                    decision.resolved = True
                    self.stats[decision.kind]['stale-generation'] += 1
                    continue
                decision.sent_at = self.clock()
                self.inflight[route] = decision
            body = dict(model=MODEL_NAME, state=decision.state, questions=decision.questions,
                        parallel=decision.kind in ('urgent', 'tactical'))
            started = self.clock()
            try:
                result = self.transport(endpoint, body)
                answers = result.get('answers', {})
                error = None if valid_answers(decision.questions, answers) else 'invalid answer'
            except Busy:
                # Another client holds the worker: retry with this state unless superseded or expired.
                with self.lock:
                    self.inflight.pop(route, None)
                    self.stats[decision.kind]['busy-retry'] += 1
                    if not decision.resolved and decision.key not in self.pending:
                        self.pending[decision.key] = decision
                time.sleep(0.005)
                continue
            except Exception as exc:  # transport, HTTP status, or JSON failure
                result, answers, error = {}, {}, f'{type(exc).__name__}: {exc}'
            finished = self.clock()
            with self.lock:
                self.inflight.pop(route, None)
                self.busy[route] += finished - started
                meta = dict(route=route, endpoint=endpoint, client_ms=round((finished - started) * 1000, 2),
                            server_ms=result.get('latency_ms'), tokens=result.get('usage', {}).get('input_tokens'),
                            queued_ms=round((started - decision.created) * 1000, 2))
                if decision.resolved:
                    self.stats[decision.kind]['late'] += 1
                    self._log(decision, 'late', answers, meta)
                elif decision.generation != self.generation():
                    decision.resolved = True
                    self.stats[decision.kind]['stale-generation'] += 1
                elif error:
                    meta['error'] = error
                    self._resolve(decision, decision.default, 'default-after-error', meta)
                else:
                    self.latency[decision.kind].append(meta['client_ms'])
                    if decision.cache_key is not None and self.memo_seconds > 0:
                        self.memo[(decision.key, decision.cache_key)] = (answers, finished)
                        if len(self.memo) > 512:
                            self.memo.pop(next(iter(self.memo)))
                    self._resolve(decision, answers, 'systemone', meta)

    def _deadlines(self):
        while True:
            with self.lock:
                if self.closed:
                    return
                now = self.clock()
                for decision in [*self.pending.values(), *self.inflight.values()]:
                    if not decision.resolved and now >= decision.deadline:
                        if decision.key in self.pending and self.pending[decision.key] is decision:
                            del self.pending[decision.key]
                        self._resolve(decision, decision.default, 'default-after-deadline',
                                      dict(waited_ms=round((now - decision.created) * 1000, 2)))
                self.lock.wait(0.005)

    # -- resolution -------------------------------------------------------
    def _resolve(self, decision, answers, provenance, meta):
        decision.resolved = True
        if answers is None:
            provenance = provenance.replace('default', 'no-default')
        self.stats[decision.kind][provenance] += 1
        self._log(decision, provenance, answers, meta)
        # Callbacks run in order on one dispatcher thread, never under the broker lock.
        self.callbacks.append((decision, answers, provenance, meta))
        self.lock.notify_all()

    def _dispatch(self):
        while True:
            with self.lock:
                while not self.callbacks and not self.closed:
                    self.lock.wait(0.05)
                if not self.callbacks and self.closed:
                    return
                item = self.callbacks.popleft()
            try:
                item[0].on_result(*item)
            except Exception as exc:
                if self.log:
                    self.log(dict(time=time.time(), key=item[0].key, callback_error=repr(exc)))

    def _log(self, decision, provenance, answers, meta):
        if self.log:
            self.log(dict(time=time.time(), key=decision.key, kind=decision.kind, provenance=provenance,
                          answers={k: v.get('choice', v.get('score', v.get('noul'))) if isinstance(v, dict) else v
                                   for k, v in (answers or {}).items()}, **meta))

    def summary(self):
        with self.lock:
            elapsed = max(1e-9, self.clock() - self.started)
            out = {}
            for kind, counts in self.stats.items():
                values = sorted(self.latency[kind])
                out[kind] = dict(counts, p50_ms=values[len(values) // 2] if values else None,
                                 p95_ms=values[min(len(values) - 1, int(len(values) * .95))] if values else None)
            out['utilization'] = {self.endpoints[r]: round(t / elapsed, 3) for r, t in self.busy.items()}
            return out

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        for thread in (*self.workers, self.timer, self.dispatcher):
            thread.join(timeout=2)
        if hasattr(self.transport, 'close'):
            self.transport.close()
