"""Sample whole-device NVIDIA VRAM without importing Torch or loading a model."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time


def sample_gpu():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'nvidia-smi failed')
    samples = []
    for line in result.stdout.splitlines():
        index, name, total, used = [part.strip() for part in line.split(',')]
        samples.append(dict(index=int(index), name=name, total_mib=int(total), used_mib=int(used)))
    return dict(time=datetime.now(timezone.utc).isoformat(), gpus=samples)


class GpuRecorder:
    def __init__(self, path, phase, interval=1):
        self.path, self.phase, self.interval = Path(path), phase, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as output:
            while not self.stop_event.is_set():
                try:
                    event = sample_gpu()
                except Exception as error:
                    event = dict(time=datetime.now(timezone.utc).isoformat(), error=str(error))
                event['phase'] = self.phase
                output.write(json.dumps(event) + '\n')
                output.flush()
                self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--seconds', type=float, default=30)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    with GpuRecorder(args.output, args.phase):
        time.sleep(args.seconds)


if __name__ == '__main__':
    main()
