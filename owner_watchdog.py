"""Exit an automatically launched Windows server when its owning application exits."""

import ctypes
import os
import threading


def configure_console():
    if os.environ.get("VON_SHOW_CONSOLE", "1" if os.name == "nt" else "0") != "1":
        return False
    if os.name != "nt":
        raise RuntimeError("The optional backend console currently requires Windows.")
    import sys

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetConsoleTitleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetConsoleWindow.restype = ctypes.c_void_p
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.IsWindowVisible.argtypes = [ctypes.c_void_p]
    window = kernel.GetConsoleWindow()
    # Double-clicking the batch launcher already gives us a visible console.
    if window and user.IsWindowVisible(window):
        kernel.SetConsoleTitleW("SystemOne Von Server")
        return True
    kernel.FreeConsole()
    if not kernel.AllocConsole():
        raise RuntimeError("Unable to allocate the requested Von console.")
    kernel.SetConsoleTitleW("SystemOne Von Server")
    # The parent can own an inherited hidden console. Reconnect Python streams
    # after detaching and allocating the backend's own console.
    sys.stdin = open("CONIN$", "r", encoding="utf-8")
    sys.stdout = open("CONOUT$", "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stderr = open("CONOUT$", "w", buffering=1, encoding="utf-8", errors="replace")
    sys.__stdin__, sys.__stdout__, sys.__stderr__ = sys.stdin, sys.stdout, sys.stderr
    return True


def start_owner_watchdog():
    value = os.environ.get("VON_OWNER_PID")
    if not value:
        return None
    try:
        owner_pid = int(value)
        if owner_pid <= 0:
            raise ValueError()
    except ValueError as error:
        raise RuntimeError("VON_OWNER_PID must identify the owning application process.") from error
    if os.name != "nt":
        raise RuntimeError("Automatic owner process monitoring currently requires Windows.")

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    # A bound process handle remains tied to this process even if its PID is reused.
    handle = kernel.OpenProcess(0x00100000, False, owner_pid)  # SYNCHRONIZE
    if not handle:
        raise RuntimeError("Cannot monitor the owning application; refusing to launch an unowned server.")

    def monitor():
        try:
            result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            os._exit(0 if result == 0 else 1)
        finally:
            kernel.CloseHandle(handle)

    thread = threading.Thread(target=monitor, name="von-owner-watchdog", daemon=True)
    thread.start()
    return thread
