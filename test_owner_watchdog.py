import os
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from owner_watchdog import configure_console, start_owner_watchdog


def test_standalone_server_has_no_owner_watchdog(monkeypatch):
    monkeypatch.delenv("VON_OWNER_PID", raising=False)
    assert start_owner_watchdog() is None


def test_hidden_console_does_not_change_console_handles(monkeypatch):
    monkeypatch.setenv("VON_SHOW_CONSOLE", "0")
    assert configure_console() is False


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_invalid_owner_fails_closed(monkeypatch, value):
    monkeypatch.setenv("VON_OWNER_PID", value)
    with pytest.raises(RuntimeError, match="owning application"):
        start_owner_watchdog()


@pytest.mark.skipif(os.name != "nt", reason="Windows process-handle lifecycle")
def test_owner_death_stops_server_before_heavy_import_finishes(tmp_path):
    # Simulate a blocked heavy import without loading Torch, weights, or a GPU.
    marker = tmp_path / "import-started"
    (tmp_path / "fastapi.py").write_text(
        "from pathlib import Path\nimport time\n"
        f"Path({str(marker)!r}).write_text('ready')\ntime.sleep(60)\n"
    )
    python = getattr(sys, "_base_executable", sys.executable)
    owner = subprocess.Popen([python, "-c", "import time; time.sleep(60)"])
    env = {**os.environ, "VON_OWNER_PID": str(owner.pid), "PYTHONPATH": str(tmp_path)}
    child = subprocess.Popen([python, str(Path(__file__).with_name("server.py"))],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 8
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists(), "Server did not reach the simulated heavy import"
        owner.kill()
        owner.wait(timeout=3)
        assert child.wait(timeout=3) == 0
    finally:
        for process in (child, owner):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        child.stderr.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows console ownership")
@pytest.mark.parametrize("show_console", [None, "1"])
def test_requested_console_allocates_a_visible_backend_window(tmp_path, show_console):
    marker = tmp_path / "console.json"
    code = (
        "from owner_watchdog import configure_console\n"
        "import ctypes,json,time\nfrom pathlib import Path\n"
        "configure_console()\n"
        "kernel=ctypes.WinDLL('kernel32'); user=ctypes.WinDLL('user32')\n"
        "kernel.GetConsoleWindow.restype=ctypes.c_void_p\n"
        "user.IsWindowVisible.argtypes=[ctypes.c_void_p]\n"
        "window=kernel.GetConsoleWindow()\n"
        "time.sleep(.2)\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'window':window,'visible':bool(user.IsWindowVisible(window))}}))\n"
        "time.sleep(30)\n"
    )
    env = dict(os.environ)
    env.pop("VON_SHOW_CONSOLE", None)
    if show_console is not None:
        env["VON_SHOW_CONSOLE"] = show_console
    child = subprocess.Popen([getattr(sys, "_base_executable", sys.executable), "-c", code],
                             cwd=Path(__file__).parent, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists(), "Backend console helper did not finish startup"
        result = json.loads(marker.read_text())
        assert result["window"] and result["visible"]
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)
