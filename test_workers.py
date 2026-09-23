"""Multi-worker server behavior with stub replicas (no model or GPU needed)."""

from model_config import MODEL_NAME

import threading
import time

from fastapi.testclient import TestClient

from server import create_app
from workers import WorkerSlots, merge_responses


class EchoBackend:
    """Answers every question with its own id; can be held to simulate a busy worker."""
    device = "test"

    def __init__(self, fail_load=False):
        self.fail_load = fail_load
        self.hold = threading.Event()
        self.hold.set()
        self.entered = threading.Event()
        self.calls = []

    def load(self):
        if self.fail_load:
            raise RuntimeError("replica failed")

    def evaluate(self, state, questions):
        self.calls.append(list(questions))
        self.entered.set()
        self.hold.wait(3)
        return {"model": MODEL_NAME,
                "answers": {q: {"type": "choice", "choice": q, "confidence": 1.0} for q in questions},
                "usage": {"input_tokens": 10 * len(questions), "output_tokens": len(questions)}}


def question(name):
    return {"type": "choice", "instructions": f"Pick {name}.", "criteria": {name: "yes", "other": "no"}}


def payload(*names, **extra):
    return {"state": "s", "questions": {n: question(n) for n in names}, **extra}


def ready(client, workers=None):
    for _ in range(200):
        body = client.get("/health").json()
        if body["ready"] and (workers is None or body["workers"]["ready"] == workers):
            return body
        time.sleep(.01)
    raise AssertionError("not ready")


def test_health_reports_workers():
    with TestClient(create_app([EchoBackend(), EchoBackend()], api_key="")) as client:
        body = ready(client, workers=2)
        assert body["workers"] == {"total": 2, "ready": 2, "busy": 0, "loading": 0, "failed": 0}


def test_concurrent_requests_use_separate_workers_and_429_only_when_all_busy():
    first, second = EchoBackend(), EchoBackend()
    for backend in (first, second):
        backend.hold.clear()
    with TestClient(create_app([first, second], api_key="")) as client:
        ready(client, workers=2)
        results = []
        threads = [threading.Thread(target=lambda: results.append(client.post("/v1/systemone", json=payload("a"))))
                   for _ in range(2)]
        for thread in threads:
            thread.start()
        assert first.entered.wait(2) and second.entered.wait(2)
        assert client.get("/health").json()["workers"]["busy"] == 2
        busy = client.post("/v1/systemone", json=payload("a"))
        assert busy.status_code == 429 and busy.headers["Retry-After"] == "1"
        first.hold.set()
        second.hold.set()
        for thread in threads:
            thread.join(3)
        assert [r.status_code for r in results] == [200, 200]


def test_questions_split_across_free_workers_keep_order_and_usage():
    first, second = EchoBackend(), EchoBackend()
    with TestClient(create_app([first, second], api_key="")) as client:
        ready(client, workers=2)
        body = client.post("/v1/systemone", json=payload("a", "b", "c")).json()
        assert list(body["answers"]) == ["a", "b", "c"]
        assert all(body["answers"][q]["choice"] == q for q in "abc")
        assert body["usage"] == {"input_tokens": 30, "output_tokens": 3}
        assert body["workers_used"] == 2
        assert sorted(first.calls + second.calls) == [["a", "c"], ["b"]]


def test_parallel_false_keeps_questions_on_one_worker():
    first, second = EchoBackend(), EchoBackend()
    with TestClient(create_app([first, second], api_key="")) as client:
        ready(client, workers=2)
        body = client.post("/v1/systemone", json=payload("a", "b", parallel=False)).json()
        assert "workers_used" not in body
        assert (first.calls + second.calls) == [["a", "b"]]


def test_batch_returns_results_in_request_order_with_per_item_errors():
    with TestClient(create_app([EchoBackend(), EchoBackend()], api_key="")) as client:
        ready(client, workers=2)
        response = client.post("/v1/systemone/batch", json={"requests": [payload("x"), payload("y"), payload("z")]})
        assert response.status_code == 200
        results = response.json()["results"]
        assert [next(iter(r["answers"])) for r in results] == ["x", "y", "z"]


def test_failed_replica_leaves_server_serving_with_fewer_workers():
    with TestClient(create_app([EchoBackend(), EchoBackend(fail_load=True)], api_key="")) as client:
        for _ in range(200):
            workers = client.get("/health").json()["workers"]
            if workers["failed"] == 1:
                break
            time.sleep(.01)
        assert workers["ready"] == 1 and workers["failed"] == 1
        assert client.post("/v1/systemone", json=payload("a", "b")).status_code == 200


def test_batch_requires_authentication():
    with TestClient(create_app([EchoBackend()], api_key="k")) as client:
        ready(client)
        assert client.post("/v1/systemone/batch", json={"requests": [payload("x")]}).status_code == 401


def test_slots_and_merge_helpers():
    slots = WorkerSlots([object(), object()])
    assert slots.acquire() == []
    slots.mark(0, "ready")
    slots.mark(1, "ready")
    assert slots.acquire(3) == [0, 1] and slots.acquire() == []
    slots.release(1)
    assert slots.acquire() == [1]
    merged = merge_responses({"b": 1, "a": 1}, [{"model": "m", "answers": {"a": 1}, "usage": {"input_tokens": 2}},
                                                 {"model": "m", "answers": {"b": 2}, "usage": {"input_tokens": 3}}])
    assert list(merged["answers"]) == ["b", "a"] and merged["usage"] == {"input_tokens": 5}
