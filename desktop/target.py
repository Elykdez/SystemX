"""Windows target identity and physical-pixel geometry."""

import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import dataclass
import os


class DesktopPause(RuntimeError):
    """A condition requiring a fresh observation or user intervention."""


user32 = ct.WinDLL("user32", use_last_error=True)
kernel32 = ct.WinDLL("kernel32", use_last_error=True)
for name, args, result in [
    ("GetForegroundWindow", [], wt.HWND),
    ("SetForegroundWindow", [wt.HWND], wt.BOOL),
    ("GetAncestor", [wt.HWND, wt.UINT], wt.HWND),
    ("WindowFromPoint", [wt.POINT], wt.HWND),
    ("IsWindow", [wt.HWND], wt.BOOL),
    ("IsWindowVisible", [wt.HWND], wt.BOOL),
    ("IsIconic", [wt.HWND], wt.BOOL),
    ("GetClientRect", [wt.HWND, ct.POINTER(wt.RECT)], wt.BOOL),
    ("ClientToScreen", [wt.HWND, ct.POINTER(wt.POINT)], wt.BOOL),
    ("GetWindowThreadProcessId", [wt.HWND, ct.POINTER(wt.DWORD)], wt.DWORD),
    ("GetWindowTextW", [wt.HWND, wt.LPWSTR, ct.c_int], ct.c_int),
    ("GetDpiForWindow", [wt.HWND], wt.UINT),
    ("GetWindowLongPtrW", [wt.HWND, ct.c_int], ct.c_ssize_t),
    ("SetProcessDpiAwarenessContext", [ct.c_void_p], wt.BOOL),
    ("GetThreadDpiAwarenessContext", [], ct.c_void_p),
    ("AreDpiAwarenessContextsEqual", [ct.c_void_p, ct.c_void_p], wt.BOOL),
]:
    fn = getattr(user32, name)
    fn.argtypes, fn.restype = args, result
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.GetProcessTimes.argtypes = [wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4
kernel32.CloseHandle.argtypes = [wt.HANDLE]


def enable_dpi():
    user32.SetProcessDpiAwarenessContext(ct.c_void_p(-4))
    if not user32.AreDpiAwarenessContextsEqual(user32.GetThreadDpiAwarenessContext(), ct.c_void_p(-4)):
        raise RuntimeError("Per-monitor DPI awareness could not be enabled. Restart the desktop player.")


def identity(hwnd):
    if not user32.IsWindow(hwnd):
        raise DesktopPause("Target window closed.")
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ct.byref(pid))
    handle = kernel32.OpenProcess(0x1000, False, pid.value)
    if not handle:
        raise DesktopPause("Cannot inspect target process. Choose a normal, non-elevated window.")
    try:
        times = [wt.FILETIME() for _ in range(4)]
        if not kernel32.GetProcessTimes(handle, *[ct.byref(t) for t in times]):
            raise DesktopPause("Cannot verify target identity.")
        return pid.value, (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel32.CloseHandle(handle)


@dataclass(frozen=True)
class Geometry:
    x: int
    y: int
    width: int
    height: int
    dpi: int


@dataclass(frozen=True)
class Target:
    hwnd: int
    pid: int
    created: int
    title: str

    @classmethod
    def attach(cls, hwnd):
        pid, created = identity(hwnd)
        title = ct.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title, len(title))
        return cls(hwnd, pid, created, title.value)

    def geometry(self):
        if identity(self.hwnd) != (self.pid, self.created):
            raise DesktopPause("Target identity changed. Select the window again.")
        if user32.IsIconic(self.hwnd) or not user32.IsWindowVisible(self.hwnd):
            raise DesktopPause("Target must be visible and not minimized.")
        rect, point = wt.RECT(), wt.POINT()
        if not user32.GetClientRect(self.hwnd, ct.byref(rect)) or not user32.ClientToScreen(self.hwnd, ct.byref(point)):
            raise DesktopPause("Cannot read target geometry.")
        if rect.right <= 0 or rect.bottom <= 0:
            raise DesktopPause("Target has no visible client area.")
        return Geometry(point.x, point.y, rect.right, rect.bottom, user32.GetDpiForWindow(self.hwnd))

    def check_point(self, x, y, foreground=False):
        if foreground and user32.GetForegroundWindow() != self.hwnd:
            raise DesktopPause("Focus changed. Focus the game, then resume with F8.")
        hit = user32.WindowFromPoint(wt.POINT(round(x), round(y)))
        if not hit or user32.GetAncestor(hit, 2) != self.hwnd:
            raise DesktopPause("The game is covered at the target cell. Uncover it and resume.")


def windows():
    found = []
    callback_type = ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    @callback_type
    def visit(hwnd, _):
        if user32.IsWindowVisible(hwnd) and not user32.GetWindowLongPtrW(hwnd, -20) & 0x80:
            try:
                target = Target.attach(hwnd)
                if target.title and target.pid != os.getpid():
                    found.append(target)
            except DesktopPause:
                pass
        return True
    user32.EnumWindows(visit, 0)
    return found
