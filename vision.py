"""Image models kept in memory while the server lives.

Addons export small TorchScript models into the server's vision folder (SYSTEMX_VISION_DIR, default
.cache/vision), for example the Chrono Divide addon's unit detector. The server loads each one at
start and again on request, on its own device, so a model shares the server's CUDA context
instead of opening one per player process, and answers in milliseconds. A model takes one uint8
image (H, W, 3) and returns rows of numbers; for a detector, [x0, y0, x1, y1, score, class]. The
JSON file beside a model describes it (name, classes, input) and is passed through unchanged.
Nothing here knows any game.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time

LOG = logging.getLogger("systemx-server")
ROOT = Path(__file__).resolve().parent
FOLDER = Path(os.environ.get("SYSTEMX_VISION_DIR", ROOT / ".cache" / "vision"))
MAX_PIXELS = 4096 * 4096


class VisionModels:
    def __init__(self, folder=FOLDER, device="cuda"):
        self.folder, self.device = Path(folder), device
        self.models = {}                 # name -> entry (module, meta, lock, statistics)
        self.lock = threading.Lock()
        self.error = None

    def load(self):
        """(Re)load every model in the folder; a model that fails is logged and left out."""
        import torch
        device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
        loaded, errors = {}, []
        for path in sorted(self.folder.glob("*.pt")) if self.folder.exists() else []:
            try:
                meta_path = path.with_suffix(".json")
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                name = str(meta.get("name", path.stem))
                before = torch.cuda.memory_allocated() if device == "cuda" else 0
                module = torch.jit.load(str(path), map_location=device).eval()
                height, width = meta.get("warmup", (736, 850))
                started = time.perf_counter()
                with torch.inference_mode():            # first call compiles kernels: pay it now
                    module(torch.zeros((height, width, 3), dtype=torch.uint8, device=device))
                loaded[name] = dict(module=module, meta=meta, lock=threading.Lock(), device=device, file=path.name,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    vram_mb=round(((torch.cuda.memory_allocated() - before) if device == "cuda" else 0) / 2 ** 20, 1),
                                    warmup_ms=round((time.perf_counter() - started) * 1000, 1), calls=0, ms=None)
                LOG.info("Vision model %s loaded on %s from %s", name, device, path.name)
            except Exception as error:                  # one bad file must not take the others down
                errors.append(f"{path.name}: {type(error).__name__}: {error}")
                LOG.exception("Vision model %s failed to load", path.name)
        with self.lock:
            self.models = loaded
            self.error = "; ".join(errors) or None
        return sorted(loaded)

    def describe(self):
        with self.lock:
            models = dict(self.models)
        return [dict(name=name, file=e["file"], sha256=e["sha256"], device=e["device"], vram_mb=e["vram_mb"],
                     warmup_ms=e["warmup_ms"], calls=e["calls"], ms=e["ms"], meta=e["meta"]) for name, e in models.items()]

    def run(self, name, data, width, height, min_score=None):
        """Rows for one image (raw uint8 bytes, height x width x 3); rows scoring under min_score (their
        fifth value) are left out. KeyError for an unknown model, ValueError for a bad image."""
        import torch
        with self.lock:
            entry = self.models[name]
        if not (0 < width and 0 < height and width * height <= MAX_PIXELS) or len(data) != width * height * 3:
            raise ValueError(f"Expected {width}x{height}x3 bytes, got {len(data)}.")
        started = time.perf_counter()
        with entry["lock"], torch.inference_mode():
            image = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(height, width, 3).to(entry["device"])
            rows = entry["module"](image).float().cpu()
        if min_score is not None and rows.ndim == 2 and rows.shape[1] >= 5:
            rows = rows[rows[:, 4] >= min_score]
        ms = (time.perf_counter() - started) * 1000
        entry["calls"] += 1
        entry["ms"] = round(ms, 2)
        return rows.tolist(), ms
