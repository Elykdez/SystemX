
from model_config import MODEL_NAME
import httpx
import pytest

from addons.minesweeper.decision import SystemOneDecision


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("addons.minesweeper.decision.time.monotonic", lambda: now[0])
    monkeypatch.setattr("addons.minesweeper.decision.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


def make_client(handler):
    return SystemOneDecision(client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)))


def health(state):
    return httpx.Response(200, json={"model": MODEL_NAME, "status": state, "ready": state == "ready"})


def test_connection_startup_loading_and_ready_have_visible_progress(clock):
    calls = []
    def respond(request):
        assert request.url.path == "/health"
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("Not started", request=request)
        return health("loading" if len(calls) == 2 else "ready")
    brain = make_client(respond)
    messages = []
    brain.wait_ready(progress=messages.append)
    assert "Waiting for server" in messages[0]
    assert "Loading AI model" in messages[1]
    assert "System One ready" in messages[2]
    assert clock[0] == 4
    brain.close()


def test_stalled_loading_times_out_with_bounded_polling(clock):
    requests = []
    def respond(request):
        requests.append(request)
        return health("loading")
    brain = make_client(respond)
    messages = []
    with pytest.raises(TimeoutError, match="Last state: Loading"):
        brain.wait_ready(timeout=8, progress=messages.append)
    assert len(requests) == 4
    assert len(messages) == 2
    assert "[6s]" in messages[-1]
    assert clock[0] == 8
    brain.close()


def test_initialization_failure_is_reported_immediately(clock):
    brain = make_client(lambda request: health("failed"))
    with pytest.raises(RuntimeError, match="initialization failed"):
        brain.wait_ready()
    assert clock[0] == 0
    brain.close()


def test_wrong_service_is_not_mistaken_for_ready(clock):
    brain = make_client(lambda request: httpx.Response(200, json={"ready": True, "status": "ready"}))
    with pytest.raises(RuntimeError, match="expected System One server"):
        brain.wait_ready()
    brain.close()


def test_ctrl_c_during_wait_propagates(clock):
    def cancelled(request):
        raise KeyboardInterrupt()
    brain = make_client(cancelled)
    with pytest.raises(KeyboardInterrupt):
        brain.wait_ready()
    brain.close()
