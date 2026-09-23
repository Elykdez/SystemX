import httpx
import pytest

from addons.minesweeper.board import Board, Box, Cell
from addons.minesweeper.decision import SystemOneDecision, choose_move
from addons.minesweeper.planner import candidates
from addons.minesweeper.session import MinesweeperSession, StaleObservation


def board(states=("hidden", "hidden"), offset=0, status="playing"):
    return Board(1, 2, tuple(Cell(0, c, state, Box(offset + c * 20, 0, 20, 20))
                             for c, state in enumerate(states)), status)


def test_systemone_receives_visible_state_and_must_select_a_candidate():
    current = board()
    options = candidates(current)
    def respond(request):
        import json
        payload = json.loads(request.content)
        assert payload["state"]["visible_board"] == ["??"]
        assert len(payload["questions"]["move"]["criteria"]) == 2
        return httpx.Response(200, json={"answers": {"move": {"choice": options[-1].id}}, "latency_ms": 1})
    brain = SystemOneDecision(client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)))
    action, details = choose_move(current, brain)
    assert action == options[-1]
    assert details["source"] == "systemone"
    brain.close()


@pytest.mark.parametrize("response", [httpx.Response(429), httpx.Response(503),
    httpx.Response(200, json={"answers": {"move": {"choice": "invented"}}})])
def test_model_failure_does_not_silently_choose_another_move(response):
    brain = SystemOneDecision(client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(lambda _: response)))
    with pytest.raises((httpx.HTTPStatusError, ValueError)):
        choose_move(board(), brain)
    brain.close()


class Driver:
    def __init__(self, observations):
        self.observations = iter(observations)
        self.clicks = []

    def observe(self):
        return next(self.observations)

    def click(self, cell, button):
        self.clicks.append((cell, button))

    def wait_for_change(self, previous):
        return self.observe()


@pytest.mark.parametrize("changed", [board(("1", "hidden")), board(offset=1)])
def test_changed_state_or_geometry_prevents_click(changed):
    driver = Driver([board(), changed])
    with pytest.raises(StaleObservation):
        MinesweeperSession(driver).step()
    assert driver.clicks == []


def test_terminal_board_never_clicks():
    driver = Driver([board(status="lost")])
    assert MinesweeperSession(driver).step()["status"] == "lost"
    assert driver.clicks == []


def test_one_confirmed_move_and_followup_observation():
    from io import StringIO
    current = board(("0", "hidden"))
    driver = Driver([current, current, board(("0", "0"), status="won")])
    log = StringIO()
    result = MinesweeperSession(driver, log=log).step()
    assert result["status"] == "won"
    assert len(driver.clicks) == 1
    assert result["decision"]["source"] == "only-candidate"
    assert '"visible_before"' in log.getvalue()
