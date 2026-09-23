"""Download the pinned model in the background without importing Torch or using a GPU."""

from owner_watchdog import start_owner_watchdog

if __name__ == "__main__":
    start_owner_watchdog()

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import traceback

from backend import FILES, MODEL_ID, MODEL_REVISION, configure_cache, download_model

CHECKPOINT_BYTES = 1581316050


def snapshot_directory():
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    hub = Path(cache) if cache else Path(os.environ["HF_HOME"]) / "hub"
    return hub / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION


def snapshot_ready(directory):
    directory = Path(directory)
    for name in FILES:
        path = directory / name
        if not path.is_file():
            return False
        if name == "option_marker.pt":
            if path.stat().st_size != CHECKPOINT_BYTES:
                return False
        else:
            try:
                with path.open(encoding="utf-8") as source:
                    if not isinstance(json.load(source), dict):
                        return False
            except (ValueError, UnicodeError):
                return False
    return True


def emit(status, **details):
    print(json.dumps({"status": status, **details}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    emit("checking")
    try:
        configure_cache()
        if snapshot_ready(snapshot_directory()):
            emit("ready")
            return 0
        if args.check_only:
            emit("missing")
            return 2
        emit("downloading")
        # Library progress and diagnostics must not contaminate the JSON protocol.
        with redirect_stdout(sys.stderr):
            directory = download_model()
        if not snapshot_ready(directory):
            raise RuntimeError("The downloaded snapshot is incomplete or has invalid files.")
        emit("ready")
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        emit("error", message="The Von model could not be prepared. Check the download log and retry.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
