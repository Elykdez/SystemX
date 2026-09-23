"""Detect physical (non-injected) mouse/keyboard input for user takeover.

Cursor-position comparison fails under browser pointer lock, so takeover uses the
injected flags of low-level hooks instead. Input from remote-desktop tools may be
marked injected and is then invisible here; the stop hotkey remains available.
"""

import ctypes as ct
from ctypes import wintypes as wt
import threading
import time

from .target import user32, kernel32

LRESULT = ct.c_ssize_t
HOOKPROC = ct.WINFUNCTYPE(LRESULT, ct.c_int, wt.WPARAM, wt.LPARAM)


class MSLLHOOKSTRUCT(ct.Structure):
    _fields_ = [('pt', wt.POINT), ('mouseData', wt.DWORD), ('flags', wt.DWORD),
                ('time', wt.DWORD), ('dwExtraInfo', ct.c_size_t)]


class KBDLLHOOKSTRUCT(ct.Structure):
    _fields_ = [('vkCode', wt.DWORD), ('scanCode', wt.DWORD), ('flags', wt.DWORD),
                ('time', wt.DWORD), ('dwExtraInfo', ct.c_size_t)]


user32.SetWindowsHookExW.argtypes = [ct.c_int, HOOKPROC, wt.HINSTANCE, wt.DWORD]
user32.SetWindowsHookExW.restype = wt.HHOOK
user32.CallNextHookEx.argtypes = [wt.HHOOK, ct.c_int, wt.WPARAM, wt.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wt.HHOOK]
user32.PostThreadMessageW.argtypes = [wt.DWORD, wt.UINT, wt.WPARAM, wt.LPARAM]
kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
kernel32.GetModuleHandleW.restype = wt.HMODULE

WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
LLMHF_INJECTED, LLKHF_INJECTED = 0x01, 0x10
WM_MOUSEMOVE = 0x0200


class PhysicalInputWatch:
    """Calls on_physical(kind) for real user input; keep the callback short.

    Tiny physical mouse jitter below `move_threshold` pixels is ignored so a desk
    bump does not pause a run; any button, wheel, or key press counts immediately.
    """

    def __init__(self, on_physical, move_threshold=12, ignore_keys=()):
        self.on_physical = on_physical
        self.move_threshold = move_threshold
        self.ignore_keys = set(ignore_keys)
        self.last_physical = None
        self.anchor = None
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='physical-input-watch', daemon=True)
        self.thread.start()
        if not self.ready.wait(3):
            raise RuntimeError('Input watch did not start.')
        if self.error:
            raise RuntimeError(self.error)

    def _mouse(self, code, wparam, lparam):
        if code == 0:
            info = ct.cast(lparam, ct.POINTER(MSLLHOOKSTRUCT)).contents
            if not info.flags & (LLMHF_INJECTED | 0x02):
                if wparam == WM_MOUSEMOVE:
                    point = (info.pt.x, info.pt.y)
                    if self.anchor is None:
                        self.anchor = point
                    elif max(abs(point[0] - self.anchor[0]), abs(point[1] - self.anchor[1])) >= self.move_threshold:
                        self._report('mouse-move')
                else:
                    self._report('mouse-button')
        return user32.CallNextHookEx(None, code, wparam, lparam)

    def _keyboard(self, code, wparam, lparam):
        if code == 0:
            info = ct.cast(lparam, ct.POINTER(KBDLLHOOKSTRUCT)).contents
            if not info.flags & LLKHF_INJECTED and info.vkCode not in self.ignore_keys:
                self._report('keyboard')
        return user32.CallNextHookEx(None, code, wparam, lparam)

    def _report(self, kind):
        self.last_physical = (kind, time.monotonic())
        try:
            self.on_physical(kind)
        except Exception:
            pass

    def reset_anchor(self):
        self.anchor = None

    def _run(self):
        self.thread_id = kernel32.GetCurrentThreadId()
        self._mouse_proc, self._keyboard_proc = HOOKPROC(self._mouse), HOOKPROC(self._keyboard)
        module = kernel32.GetModuleHandleW(None)
        hooks = [user32.SetWindowsHookExW(WH_MOUSE_LL, self._mouse_proc, module, 0),
                 user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._keyboard_proc, module, 0)]
        if not all(hooks):
            self.error = 'Cannot install input hooks for takeover detection.'
            for hook in filter(None, hooks):
                user32.UnhookWindowsHookEx(hook)
            self.ready.set()
            return
        self.ready.set()
        try:
            message = wt.MSG()
            while user32.GetMessageW(ct.byref(message), None, 0, 0) > 0:
                pass
        finally:
            for hook in hooks:
                user32.UnhookWindowsHookEx(hook)

    def close(self):
        user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
        self.thread.join(timeout=2)


# -- Raw Input: preferred. Injected input has hDevice == 0; SetCursorPos produces none. ------

WNDPROC = ct.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)


class WNDCLASSEXW(ct.Structure):
    _fields_ = [('cbSize', wt.UINT), ('style', wt.UINT), ('lpfnWndProc', WNDPROC), ('cbClsExtra', ct.c_int),
                ('cbWndExtra', ct.c_int), ('hInstance', wt.HINSTANCE), ('hIcon', wt.HICON),
                ('hCursor', wt.HANDLE), ('hbrBackground', wt.HBRUSH), ('lpszMenuName', wt.LPCWSTR),
                ('lpszClassName', wt.LPCWSTR), ('hIconSm', wt.HICON)]


class RAWINPUTDEVICE(ct.Structure):
    _fields_ = [('usUsagePage', wt.USHORT), ('usUsage', wt.USHORT), ('dwFlags', wt.DWORD), ('hwndTarget', wt.HWND)]


class RAWINPUTHEADER(ct.Structure):
    _fields_ = [('dwType', wt.DWORD), ('dwSize', wt.DWORD), ('hDevice', wt.HANDLE), ('wParam', wt.WPARAM)]


class RAWMOUSE(ct.Structure):
    # usButtonFlags/usButtonData overlay a ULONG union aligned at offset 4.
    _fields_ = [('usFlags', wt.USHORT), ('_padding', wt.USHORT), ('usButtonFlags', wt.USHORT),
                ('usButtonData', wt.USHORT), ('ulRawButtons', wt.ULONG), ('lLastX', wt.LONG),
                ('lLastY', wt.LONG), ('ulExtraInformation', wt.ULONG)]


class RAWKEYBOARD(ct.Structure):
    _fields_ = [('MakeCode', wt.USHORT), ('Flags', wt.USHORT), ('Reserved', wt.USHORT), ('VKey', wt.USHORT),
                ('Message', wt.UINT), ('ExtraInformation', wt.ULONG)]


class RAWDATA(ct.Union):
    _fields_ = [('mouse', RAWMOUSE), ('keyboard', RAWKEYBOARD)]


class RAWINPUT(ct.Structure):
    _fields_ = [('header', RAWINPUTHEADER), ('data', RAWDATA)]


user32.RegisterClassExW.argtypes = [ct.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype = wt.ATOM
user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ct.c_int, ct.c_int, ct.c_int,
                                   ct.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID]
user32.CreateWindowExW.restype = wt.HWND
user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterRawInputDevices.argtypes = [ct.POINTER(RAWINPUTDEVICE), wt.UINT, wt.UINT]
user32.GetRawInputData.argtypes = [wt.HANDLE, wt.UINT, wt.LPVOID, ct.POINTER(wt.UINT), wt.UINT]
user32.GetRawInputData.restype = wt.UINT
user32.DestroyWindow.argtypes = [wt.HWND]
user32.DispatchMessageW.argtypes = [ct.POINTER(wt.MSG)]

WM_INPUT, RID_INPUT, RIDEV_INPUTSINK, RIDEV_REMOVE = 0x00FF, 0x10000003, 0x00000100, 0x00000001
RIM_TYPEMOUSE, RIM_TYPEKEYBOARD = 0, 1


class RawInputWatch:
    """Physical input detector based on Raw Input device handles.

    Mouse motion must accumulate `move_threshold` raw counts (a desk bump is ignored);
    any physical button, wheel, or key (except `ignore_keys`) reports at once.
    """

    def __init__(self, on_physical, move_threshold=12, ignore_keys=()):
        self.on_physical = on_physical
        self.move_threshold = move_threshold
        self.ignore_keys = set(ignore_keys)
        self.motion = 0
        self.last_physical = None
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='raw-input-watch', daemon=True)
        self.thread.start()
        if not self.ready.wait(3):
            raise RuntimeError('Raw input watch did not start.')
        if self.error:
            raise RuntimeError(self.error)

    def reset_anchor(self):
        self.motion = 0

    def _report(self, kind):
        self.last_physical = (kind, time.monotonic())
        try:
            self.on_physical(kind)
        except Exception:
            pass

    def _handle(self, lparam):
        size = wt.UINT(0)
        header = ct.sizeof(RAWINPUTHEADER)
        user32.GetRawInputData(lparam, RID_INPUT, None, ct.byref(size), header)
        buffer = ct.create_string_buffer(max(size.value, ct.sizeof(RAWINPUT)))
        if user32.GetRawInputData(lparam, RID_INPUT, buffer, ct.byref(size), header) == 0xFFFFFFFF:
            return
        self.process(ct.cast(buffer, ct.POINTER(RAWINPUT)).contents)

    def process(self, raw):
        """Classify one raw input record; injected input (SendInput) has no device handle."""
        if not raw.header.hDevice:
            return
        if raw.header.dwType == RIM_TYPEMOUSE:
            mouse = raw.data.mouse
            if mouse.usButtonFlags:
                self._report('mouse-button')
            elif not mouse.usFlags & 0x01:  # relative motion from a real device
                self.motion += abs(mouse.lLastX) + abs(mouse.lLastY)
                if self.motion >= self.move_threshold:
                    self.motion = 0
                    self._report('mouse-move')
        elif raw.header.dwType == RIM_TYPEKEYBOARD:
            keyboard = raw.data.keyboard
            if keyboard.VKey not in self.ignore_keys and not keyboard.Flags & 0x01:   # key down only
                self._report('keyboard')

    def _proc(self, hwnd, message, wparam, lparam):
        if message == WM_INPUT:
            try:
                self._handle(lparam)
            except Exception:
                pass
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def _run(self):
        self.thread_id = kernel32.GetCurrentThreadId()
        module = kernel32.GetModuleHandleW(None)
        self._wndproc = WNDPROC(self._proc)
        name = f'SystemOneRawInput{id(self)}'
        wc = WNDCLASSEXW(ct.sizeof(WNDCLASSEXW), 0, self._wndproc, 0, 0, module, None, None, None, None, name, None)
        if not user32.RegisterClassExW(ct.byref(wc)):
            self.error = 'Cannot register the raw input window class.'
            self.ready.set()
            return
        hwnd = user32.CreateWindowExW(0, name, None, 0, 0, 0, 0, 0, wt.HWND(-3), None, module, None)
        devices = (RAWINPUTDEVICE * 2)(RAWINPUTDEVICE(1, 2, RIDEV_INPUTSINK, hwnd),
                                       RAWINPUTDEVICE(1, 6, RIDEV_INPUTSINK, hwnd))
        if not hwnd or not user32.RegisterRawInputDevices(devices, 2, ct.sizeof(RAWINPUTDEVICE)):
            self.error = 'Cannot register raw input devices for takeover detection.'
            self.ready.set()
            return
        self.ready.set()
        try:
            message = wt.MSG()
            while user32.GetMessageW(ct.byref(message), None, 0, 0) > 0:
                user32.DispatchMessageW(ct.byref(message))
        finally:
            remove = (RAWINPUTDEVICE * 2)(RAWINPUTDEVICE(1, 2, RIDEV_REMOVE, None),
                                          RAWINPUTDEVICE(1, 6, RIDEV_REMOVE, None))
            user32.RegisterRawInputDevices(remove, 2, ct.sizeof(RAWINPUTDEVICE))
            user32.DestroyWindow(hwnd)

    def close(self):
        user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
        self.thread.join(timeout=2)


def watch_physical_input(on_physical, **kwargs):
    """Raw Input when available (robust under pointer lock), else low-level hooks."""
    try:
        return RawInputWatch(on_physical, **kwargs)
    except RuntimeError:
        return PhysicalInputWatch(on_physical, **kwargs)
