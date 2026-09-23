"""Runs the unit detector (detector_net.py, trained by tools/detector.py) on the tactical view.

The model lives in the SystemX server when one runs on this machine: the player copies it into
the server's vision folder once and asks it to load it (vision.py), and from then on it stays in
memory as long as the server does, on the server's GPU and in its CUDA context (a player process
would need a context of its own). Without a server the player runs it itself, on the CPU, and
looks for a server again now and then. The model finds units by their shape; the house color
inside a box says whose unit it is.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import colors

MODEL = Path(__file__).resolve().parents[1] / 'profiles' / 'detector-units.pt'
CLASSES = ('infantry', 'vehicle')
RETRY = 20.0                     # seconds between looks for a server while running locally
LOOPBACK = ('http://127.0.0.1', 'http://localhost')


@dataclass(frozen=True)
class Detection:
    box: tuple                   # (x0, y0, x1, y1) canvas pixels
    side: str                    # 'ours' or 'enemy', by the house color inside the box
    kind: str                    # 'infantry' or 'vehicle'
    score: float


def side_of(canvas, box, share=0.03, compressed=False):
    """Whose unit the box holds: the house color with the most pixels in it (at least `share` of
    the box), or None when neither shows (then it is nobody's unit we know)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    crop = canvas[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if crop.size == 0:
        return None
    own = int(colors.OWN.mask(crop, compressed=compressed).sum())
    enemy = int(colors.enemy_mask(crop, compressed=compressed).sum())
    if max(own, enemy) < share * crop.shape[0] * crop.shape[1]:
        return None
    return 'ours' if own >= enemy else 'enemy'


class Detector:
    def __init__(self, path=MODEL, server='http://127.0.0.1:8000', threshold=None, clock=time.monotonic):
        self.path = Path(path)
        self.meta = json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8'))
        self.name = self.meta.get('name', self.path.stem)
        self.threshold = float(threshold if threshold is not None else self.meta.get('threshold', 0.4))
        self.server = server if server and server.startswith(LOOPBACK) else None
        self.clock = clock
        self.client = self.model = None
        self.where = None                # 'server' | 'local' | None (not connected yet)
        self.error = None
        self.last_try = -1e9
        self.ms = None

    @staticmethod
    def available(path=MODEL):
        return Path(path).exists() and Path(path).with_suffix('.json').exists()

    def detect(self, canvas, region):
        """Units in the region (x, y, w, h) of the canvas, as Detections with canvas boxes."""
        x, y, w, h = region
        view = np.ascontiguousarray(canvas[y:y + h, x:x + w])
        started = time.perf_counter()
        rows = self._rows(view)
        self.ms = round((time.perf_counter() - started) * 1000, 1)
        found = []
        for x0, y0, x1, y1, score, cls in rows:
            if score < self.threshold:
                continue
            box = (int(x0) + x, int(y0) + y, int(x1) + x, int(y1) + y)
            side = side_of(canvas, box)
            if side is not None:
                found.append(Detection(box, side, CLASSES[int(cls)], round(float(score), 3)))
        return found

    def _rows(self, view):
        if self.where != 'server' and self.clock() - self.last_try >= RETRY:
            self._use_server()
        if self.where == 'server':
            try:
                return self.client.detect(self.name, view, min_score=self.threshold)
            except Exception as error:                        # the server went away: run it here
                self.error, self.where = f'server: {type(error).__name__}: {error}', None
        return self._local(view)

    def _use_server(self):
        """Have the local server keep the model: copy it into its vision folder unless it already
        holds this very model (same checksum), then load it there."""
        self.last_try = self.clock()
        if self.server is None:
            return
        try:
            from systemone_client import SystemOneClient
            client = SystemOneClient(self.server, timeout=5)
            info = client.vision()
            loaded = {m['name']: m for m in info.get('models', [])}
            if loaded.get(self.name, {}).get('sha256') != self.meta.get('sha256'):
                folder = Path(info['folder'])
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.path, folder / self.path.name)
                shutil.copyfile(self.path.with_suffix('.json'), folder / self.path.with_suffix('.json').name)
                info = client.vision_reload()
                loaded = {m['name']: m for m in info.get('models', [])}
            if self.name not in loaded:
                raise RuntimeError(info.get('error') or 'the server did not load the model')
            self.client, self.where, self.error = client, 'server', None
            self.model = None                                 # free the local copy
        except Exception as error:
            self.error = f'server: {type(error).__name__}: {error}'

    def _local(self, view):
        if self.model is None:
            try:
                import torch
                torch.set_num_threads(2)                      # leave the CPU to the game and the capture
                self.model = torch.jit.load(str(self.path), map_location='cpu').eval()
                self.where = 'local'
            except Exception as error:
                self.error = f'local: {type(error).__name__}: {error}'
                return []
        import torch
        with torch.inference_mode():
            return self.model(torch.from_numpy(view)).numpy().tolist()

    def status(self):
        return dict(where=self.where, ms=self.ms, error=self.error, name=self.name,
                    trained=self.meta.get('trained'), evaluation=self.meta.get('evaluation'))
