"""Opt-in real-model test server with CUDA allocator measurements; core files stay unchanged."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from backend import VonBackend, configure_cache
from server import create_app


class MeasuredBackend(VonBackend):
    def __init__(self, output):
        super().__init__('cuda')
        self.output = output
        self.phase = 'warmup'

    def record(self):
        import torch
        torch.cuda.synchronize()
        event = dict(time=datetime.now(timezone.utc).isoformat(), phase=self.phase,
                     allocated_mib=torch.cuda.memory_allocated() / 1048576,
                     reserved_mib=torch.cuda.memory_reserved() / 1048576,
                     peak_allocated_mib=torch.cuda.max_memory_allocated() / 1048576,
                     peak_reserved_mib=torch.cuda.max_memory_reserved() / 1048576)
        with self.output.open('a', encoding='utf-8') as log:
            log.write(json.dumps(event) + '\n')

    def load(self):
        super().load()
        self.phase = 'ready'
        self.record()
        self.phase = 'inference'

    def evaluate(self, state, questions):
        result = super().evaluate(state, questions)
        self.record()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configure_cache()
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    uvicorn.run(create_app(MeasuredBackend(args.output)), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
