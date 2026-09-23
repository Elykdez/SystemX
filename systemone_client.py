"""Small thread-safe client for SystemX: decisions, parallel splitting, batches, worker status.

    client = SystemOneClient("http://127.0.0.1:8000")
    client.wait_ready()
    answer = client.decide(state, questions)                  # questions split across free workers
    results = client.batch([(state_a, qs_a), (state_b, qs_b)])  # independent decisions in parallel
    client.workers()                                          # {"total", "ready", "busy", ...}
    rows = client.detect("ra2-units", image, min_score=0.4)   # an image model kept in the server

A 429 means every worker is busy: retry with fresh state rather than queueing old decisions.
"""

from model_config import MODEL_NAME

import os
import time

import httpx
import numpy as np

MODEL = MODEL_NAME


class Busy(RuntimeError):
    """All server workers were busy; the caller decides whether to retry with fresh state."""


class SystemOneClient:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=30.0, api_key=None):
        api_key = api_key if api_key is not None else os.environ.get("SYSTEMX_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def health(self):
        response = self.http.get("/health", timeout=3)
        response.raise_for_status()
        return response.json()

    def workers(self):
        """Worker counts; servers older than the parallel release report one worker."""
        return self.health().get("workers", {"total": 1, "ready": 1, "busy": 0})

    def wait_ready(self, timeout=300):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = self.health()
                if status.get("status") == "failed":
                    raise RuntimeError("System One initialization failed; inspect the server console.")
                if status.get("ready"):
                    return status
            except httpx.TransportError:
                pass
            time.sleep(1)
        raise TimeoutError("System One server did not become ready.")

    def decide(self, state, questions, parallel=True, timeout=None):
        body = {"model": MODEL, "state": state, "questions": questions, "parallel": parallel}
        response = self.http.post("/v1/systemone", json=body, timeout=timeout or self.http.timeout)
        if response.status_code == 429:
            raise Busy(response.json().get("detail", "busy"))
        response.raise_for_status()
        return response.json()

    def vision(self):
        """The image models the server keeps in memory, and the folder it loads them from."""
        response = self.http.get("/v1/vision/models", timeout=5)
        response.raise_for_status()
        return response.json()

    def vision_reload(self):
        response = self.http.post("/v1/vision/reload", timeout=60)
        response.raise_for_status()
        return response.json()

    def detect(self, model, image, min_score=None, timeout=5.0):
        """Rows from an image model for one uint8 image (H, W, 3), sent as raw bytes."""
        image = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = image.shape[:2]
        params = {"model": model, "width": width, "height": height}
        if min_score is not None:
            params["min_score"] = min_score
        response = self.http.post("/v1/vision/detect", params=params, content=image.tobytes(),
                                  headers={"Content-Type": "application/octet-stream"}, timeout=timeout)
        response.raise_for_status()
        return response.json()["rows"]

    def batch(self, items, timeout=None):
        """items: iterable of (state, questions); returns responses or {'error': ...} per item."""
        body = {"requests": [{"model": MODEL, "state": s, "questions": q} for s, q in items]}
        response = self.http.post("/v1/systemone/batch", json=body, timeout=timeout or self.http.timeout)
        response.raise_for_status()
        return response.json()["results"]
