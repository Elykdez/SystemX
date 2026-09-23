"""Image models kept in the server's memory (vision.py) and their HTTP routes."""

import json
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from server import create_app  # noqa: E402
from test_server import StubBackend, wait_ready  # noqa: E402
from vision import VisionModels  # noqa: E402


class Brightness(torch.nn.Module):
    """A stand-in detector: one box scored by the image's mean brightness, one weak box."""

    def forward(self, image):
        score = image.float().mean() / 255.0
        return torch.stack([torch.stack([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(11.0),
                                         torch.tensor(12.0), score, torch.tensor(1.0)]),
                            torch.tensor([5.0, 5.0, 6.0, 6.0, 0.05, 0.0])])


@pytest.fixture()
def folder(tmp_path):
    torch.jit.save(torch.jit.script(Brightness()), str(tmp_path / "bright.pt"))
    (tmp_path / "bright.json").write_text(json.dumps({"name": "bright", "classes": ["a", "b"], "warmup": [8, 8]}))
    (tmp_path / "broken.pt").write_bytes(b"not a model")
    return tmp_path


def test_models_load_once_and_answer(folder):
    models = VisionModels(folder, device="cpu")
    assert models.load() == ["bright"] and "broken.pt" in models.error        # a bad file is left out
    image = np.full((4, 5, 3), 255, np.uint8).tobytes()
    rows, ms = models.run("bright", image, 5, 4)
    assert len(rows) == 2 and rows[0][4] == pytest.approx(1.0) and ms >= 0
    rows, _ = models.run("bright", image, 5, 4, min_score=0.5)
    assert len(rows) == 1
    described = models.describe()[0]
    assert described["calls"] == 2 and described["meta"]["classes"] == ["a", "b"] and len(described["sha256"]) == 64
    with pytest.raises(ValueError):
        models.run("bright", image[:-1], 5, 4)
    with pytest.raises(KeyError):
        models.run("missing", image, 5, 4)


def test_vision_routes(folder):
    models = VisionModels(folder, device="cpu")
    with TestClient(create_app(StubBackend(), api_key="k", vision=models)) as client:
        wait_ready(client)
        for _ in range(200):                                   # loaded at start, after System One
            if client.get("/health").json().get("vision"):
                break
            time.sleep(0.01)
        auth = {"Authorization": "Bearer k"}
        assert client.get("/v1/vision/models").status_code == 401
        listed = client.get("/v1/vision/models", headers=auth).json()
        assert listed["folder"] == str(folder) and [m["name"] for m in listed["models"]] == ["bright"]
        image = np.zeros((600, 800, 3), np.uint8)              # 1.4 MB: over the decision limit, under this one
        response = client.post("/v1/vision/detect", params=dict(model="bright", width=800, height=600, min_score=0.01),
                               content=image.tobytes(), headers=auth)
        assert response.status_code == 200 and response.json()["rows"] == [[5.0, 5.0, 6.0, 6.0, pytest.approx(0.05), 0.0]]
        wrong = client.post("/v1/vision/detect", params=dict(model="bright", width=10, height=10), content=b"x", headers=auth)
        assert wrong.status_code == 422
        missing = client.post("/v1/vision/detect", params=dict(model="nope", width=1, height=1), content=b"xyz", headers=auth)
        assert missing.status_code == 404
        assert client.post("/v1/vision/reload", headers=auth).json()["models"][0]["name"] == "bright"


def test_a_server_without_image_models_says_so():
    with TestClient(create_app(StubBackend())) as client:
        wait_ready(client)
        assert client.get("/v1/vision/models").status_code == 404
