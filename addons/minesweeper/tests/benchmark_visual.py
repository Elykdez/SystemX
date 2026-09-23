"""Benchmark saved visual observations and optional real System One requests; never sends input."""

import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import cv2

from desktop.capture import Frame
from desktop.input import Gate
from desktop.target import Geometry
from addons.minesweeper.desktop_driver import DesktopDriver
from addons.minesweeper.decision import SystemOneDecision, choose_move
from addons.minesweeper.visual.recognition import Grid, Profile, recognize


def distribution(values):
    ordered = sorted(values)
    return dict(count=len(values), min=round(ordered[0], 3),
                median=round(statistics.median(values), 3),
                p95=round(ordered[min(len(values)-1, int(len(values)*.95))], 3),
                max=round(ordered[-1], 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True, help='Folder containing step-N.png screenshots')
    parser.add_argument('--grid', nargs=4, type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', help='Optionally measure real requests using a persistent HTTP client')
    args = parser.parse_args()
    profile = Profile.load(Path(__file__).parents[1] / 'visual/profiles/classic-web.json')
    grid = Grid(*args.grid, 9, 9, 10)
    images = [cv2.imread(str(p)) for p in sorted(args.folder.glob('step-*.png'),
                                               key=lambda p: int(p.stem.split('-')[1]))]
    if not images or any(image is None for image in images):
        parser.error('Readable step-N.png screenshots are required')
    uncached, cached, decisions, server_times, cycles, observations, captures, non_model = [], [], [], [], [], [], [], []
    # Clearing before each cell recreates the original recognition path.
    classify = profile.classify
    def without_cache(image):
        profile._classifications.clear()
        return classify(image)
    profile.classify = without_cache
    for image in images:
        start = time.perf_counter()
        recognize(image, grid, profile)
        uncached.append((time.perf_counter() - start) * 1000)
    profile.classify = classify
    profile._classifications.clear()
    brain = SystemOneDecision(args.server) if args.server else None
    try:
        for image in images:
            start = time.perf_counter()
            board, details = recognize(image, grid, profile)
            cached.append((time.perf_counter() - start) * 1000)
            if board is None:
                raise ValueError(f'Unknown cells in benchmark: {details}')
            if board.status != 'playing':
                continue
            geometry = Geometry(0, 0, image.shape[1], image.shape[0], 96)
            reads = [0]
            def read(region):
                reads[0] += 1
                x, y, width, height = region
                return Frame(reads[0], time.monotonic(), geometry,
                             image[y:y + height, x:x + width].copy(), (x, y))
            capture = SimpleNamespace(target=SimpleNamespace(check_point=lambda *a, **kw: None), read=read)
            gate = Gate()
            gate.start()
            driver = DesktopDriver(capture, None, grid, profile, gate, gate.generation, geometry)
            start = time.perf_counter()
            driver.observe()
            decision_start = time.perf_counter()
            _, decision = choose_move(board, brain)
            decisions.append((time.perf_counter() - decision_start) * 1000)
            if decision.get('latency_ms') is not None:
                server_times.append(decision['latency_ms'])
                non_model.append(max(0, decisions[-1] - decision['latency_ms']))
            # The session observes initially, after deciding, before input, and after input.
            driver.observe()
            driver.observe()
            # An input invalidates stability history, even when replaying static pixels.
            driver._previous_observation = None
            driver.observe()
            cycles.append((time.perf_counter() - start) * 1000)
            observations.append(cycles[-1] - decisions[-1])
            captures.append(reads[0])
    finally:
        if brain:
            brain.close()
    report = dict(uncached_recognition_ms=distribution(uncached),
                  cached_recognition_ms=distribution(cached),
                  decision_wall_ms=distribution(decisions),
                  replay_cycle_ms=distribution(cycles),
                  observation_wall_ms=distribution(observations),
                  captures_per_cycle=distribution(captures),
                  non_model_decision_ms=distribution(non_model) if non_model else None,
                  model_server_ms=distribution(server_times) if server_times else None,
                  limitations='Saved pixels with simulated capture and hit tests. No OS input, live capture, UI rendering, or game render latency. Replay cycles are not measured live clicks per second.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
