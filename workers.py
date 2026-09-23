"""Parallel System One workers: several model replicas behind one HTTP endpoint.

The primary replica runs in the server process. Extra replicas each run in their
own process with a private CUDA context, so their inferences overlap on the GPU
instead of serializing behind one Python interpreter. Replicas exit when the
server exits (owner watchdog) or when their pipe closes.
"""

import logging
import multiprocessing
import os
import threading

LOG = logging.getLogger("systemx-server")


def default_workers(device):
    """Up to three int8 replicas on 24 GB GPUs; one on smaller GPUs/CPU."""
    value = os.environ.get("SYSTEMX_WORKERS")
    if value:
        count = int(value)
        if not 1 <= count <= 3:
            raise ValueError("SYSTEMX_WORKERS must be 1-3.")
        return count
    if device == "cuda":
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 23 * 2 ** 30:
            return 3 if os.environ.get("SYSTEMX_QUANTIZATION", "int8") == "int8" else 2
    return 1


def _worker_main(connection, device, max_tokens, owner_pid):
    os.environ["SYSTEMX_OWNER_PID"] = str(owner_pid)
    from owner_watchdog import start_owner_watchdog
    from backend import ModelBackend, configure_cache

    start_owner_watchdog()
    configure_cache()
    try:
        backend = ModelBackend(device, max_tokens)
        backend.load()
    except Exception as error:  # report, then exit; the server keeps its other workers
        connection.send(("error", f"{type(error).__name__}: {error}"))
        return
    connection.send(("ready", None))
    while True:
        try:
            item = connection.recv()
        except (EOFError, OSError):
            return
        if item is None:
            return
        try:
            connection.send(("ok", backend.evaluate(*item)))
        except ValueError as error:
            connection.send(("value_error", str(error)))
        except Exception as error:
            connection.send(("error", f"{type(error).__name__}: {error}"))


class ProcessBackend:
    """A ModelBackend replica in a child process with the same load/evaluate interface."""

    def __init__(self, device="cuda", max_tokens=2048):
        self.device, self.max_tokens = device, max_tokens
        self.process = self.connection = None
        self.lock = threading.Lock()

    def load(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker_main, name="systemx-worker", daemon=True,
                                       args=(child, self.device, self.max_tokens, os.getpid()))
        self.process.start()
        child.close()
        kind, detail = self.connection.recv()
        if kind != "ready":
            self.close()
            raise RuntimeError(f"System One worker failed to load: {detail}")

    def evaluate(self, state, questions):
        with self.lock:
            try:
                self.connection.send((state, questions))
                kind, value = self.connection.recv()
            except (EOFError, OSError) as error:
                raise RuntimeError("System One worker process is gone.") from error
        if kind == "ok":
            return value
        if kind == "value_error":
            raise ValueError(value)
        raise RuntimeError(value)

    def close(self):
        if self.connection is not None:
            try:
                self.connection.send(None)
            except (OSError, ValueError):
                pass
        if self.process is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()


class WorkerSlots:
    """Tracks which loaded replicas are free. Never queues work implicitly."""

    def __init__(self, engines):
        self.engines = list(engines)
        self.state = ["loading"] * len(self.engines)   # loading | ready | busy | failed
        self.condition = threading.Condition()

    def mark(self, index, state):
        with self.condition:
            self.state[index] = state
            self.condition.notify_all()

    def acquire(self, count=1, timeout=0.0):
        """Take up to `count` free replicas; wait up to `timeout` for the first one."""
        with self.condition:
            if timeout:
                self.condition.wait_for(lambda: "ready" in self.state, timeout)
            taken = [i for i, s in enumerate(self.state) if s == "ready"][:count]
            for index in taken:
                self.state[index] = "busy"
            return taken

    def release(self, index):
        self.mark(index, "ready")

    def status(self):
        with self.condition:
            return dict(total=len(self.state), ready=self.state.count("ready") + self.state.count("busy"),
                        busy=self.state.count("busy"), loading=self.state.count("loading"),
                        failed=self.state.count("failed"))

    def close(self):
        for engine in self.engines:
            if hasattr(engine, "close"):
                try:
                    engine.close()
                except Exception:
                    LOG.exception("Worker shutdown failed")


def merge_responses(questions, parts):
    """Combine per-question responses in the request's question order."""
    answers, usage = {}, {}
    for part in parts:
        answers.update(part["answers"])
        for name, value in part.get("usage", {}).items():
            usage[name] = usage.get(name, 0) + value
    merged = dict(parts[0])
    merged["answers"] = {qid: answers[qid] for qid in questions}
    merged["usage"] = usage
    return merged
