"""HTTP boundary tests. Stub is test-only; production has no fallback backend."""

from model_config import MODEL_DESCRIPTION, MODEL_NAME, MODEL_OWNER, MODEL_REVISION

import threading
import time

from fastapi.testclient import TestClient
import pytest

from server import create_app


class StubBackend:
    device = "test"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.error = None

    def load(self):
        pass

    def evaluate(self, state, questions):
        self.entered.set()
        self.release.wait(3)
        if self.error:
            raise self.error
        return {"model": MODEL_NAME, "answers": {
            "action": {"type": "choice", "choice": "hold", "confidence": .6,
                       "probabilities": {"hold": .8, "attack": .2}}}, "usage": {"input_tokens": 12, "output_tokens": 1}}


def payload():
    return {"state": "Our army is outnumbered.", "questions": {"action": {
        "type": "choice", "instructions": "Select a legal action.",
        "criteria": {"hold": "Defend our planet.", "attack": "Attack the enemy planet."}}}}


def wait_ready(client):
    for _ in range(100):
        if client.get("/ready").status_code == 200:
            return
        time.sleep(.01)
    raise AssertionError("Server never became ready")


def test_real_wire_shape_and_authentication():
    with TestClient(create_app(StubBackend(), api_key="test-key")) as client:
        wait_ready(client)
        assert client.post("/v1/systemone", json=payload()).status_code == 401
        response = client.post("/v1/systemone", json=payload(), headers={"Authorization": "Bearer test-key"})
        assert response.status_code == 200
        assert response.json()["answers"]["action"]["choice"] == "hold"
        assert response.json()["latency_ms"] >= 0


def test_health_and_model_listing_use_the_shared_model_configuration():
    with TestClient(create_app(StubBackend(), api_key="")) as client:
        wait_ready(client)
        health = client.get("/health").json()
        assert health["model"] == MODEL_NAME and health["revision"] == MODEL_REVISION
        listing = client.get("/v1/models").json()
        assert listing["models"] == [{"name": MODEL_NAME, "description": MODEL_DESCRIPTION}]
        assert listing["data"] == [{"id": MODEL_NAME, "object": "model", "owned_by": MODEL_OWNER}]


@pytest.mark.parametrize("question", [
    {"type": "choice", "instructions": "Choose.", "criteria": {}},
    {"type": "choice", "instructions": "Choose.", "criteria": {str(i): "Option" for i in range(17)}},
    {"type": "score", "instructions": "Rate.", "criteria": ["Only one"]},
    {"type": "score", "instructions": "Rate.", "criteria": {"a": "bad", "b": "good"}},
    {"type": "noul", "instructions": "Is true?", "criteria": {"yes": "Wrong key"}},
    {"type": "unknown", "instructions": "What?"},
])
def test_invalid_questions_are_rejected(question):
    with TestClient(create_app(StubBackend(), api_key="")) as client:
        assert client.post("/v1/systemone", json={"state": "test", "questions": {"q": question}}).status_code == 422


@pytest.mark.parametrize("model", ["unknown-model", "",
                                   MODEL_NAME + "-unsupported"])
def test_bounded_body_and_unknown_model(model):
    with TestClient(create_app(StubBackend(), api_key="")) as client:
        data = payload()
        data["model"] = model
        assert client.post("/v1/systemone", json=data).status_code == 422
        assert client.post("/v1/systemone", content=b"x" * 65537).status_code == 413


def test_busy_does_not_queue_old_decisions():
    backend = StubBackend()
    backend.release.clear()
    with TestClient(create_app(backend, api_key="")) as client:
        wait_ready(client)
        results = []
        worker = threading.Thread(target=lambda: results.append(client.post("/v1/systemone", json=payload())))
        worker.start()
        assert backend.entered.wait(2)
        response = client.post("/v1/systemone", json=payload())
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "1"
        backend.release.set()
        worker.join(3)
        assert results[0].status_code == 200


def test_failed_load_remains_unavailable():
    class BrokenBackend(StubBackend):
        def load(self):
            raise RuntimeError("broken weights")

    with TestClient(create_app(BrokenBackend(), api_key="")) as client:
        for _ in range(100):
            if client.get("/health").json()["status"] == "failed":
                break
            time.sleep(.01)
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/systemone", json=payload()).status_code == 503


def test_input_error_and_inference_failure_release_lock():
    backend = StubBackend()
    with TestClient(create_app(backend, api_key="")) as client:
        wait_ready(client)
        backend.error = ValueError("Token limit exceeded")
        assert client.post("/v1/systemone", json=payload()).status_code == 422
        backend.error = RuntimeError("device failed")
        assert client.post("/v1/systemone", json=payload()).status_code == 503
        backend.error = None
        assert client.post("/v1/systemone", json=payload()).status_code == 200


def test_shutdown_requires_ownership_token_and_loopback():
    calls = []
    app = create_app(StubBackend(), api_key="", management_token="owner-secret",
                     shutdown_callback=lambda: calls.append("stopped"))
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        assert client.post("/internal/shutdown").status_code == 401
        assert client.post("/internal/shutdown", headers={"X-SystemX-Management-Token": "wrong"}).status_code == 401
        assert calls == []
        assert client.post("/internal/shutdown", headers={"X-SystemX-Management-Token": "owner-secret"}).status_code == 202
        assert calls == ["stopped"]
    with TestClient(app, client=("192.0.2.1", 12345)) as client:
        assert client.post("/internal/shutdown", headers={"X-SystemX-Management-Token": "owner-secret"}).status_code == 403
        assert calls == ["stopped"]


def test_manual_server_has_no_management_endpoint():
    with TestClient(create_app(StubBackend(), api_key="")) as client:
        assert client.post("/internal/shutdown", headers={"X-SystemX-Management-Token": "anything"}).status_code == 404
