"""The shared inspector identifies itself to the Windows shell, not as Python."""

import ctypes
import sys

import pytest

if sys.platform != 'win32':
    pytest.skip('Windows taskbar integration', allow_module_level=True)

import tkinter as tk

from desktop.inspector import Inspector


def test_inspector_sets_windows_identity_and_icon(capfd):
    with capfd.disabled():
        root = tk.Tk()
    root.withdraw()
    try:
        root.update_idletasks()
        user32 = ctypes.WinDLL('user32')
        user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
        user32.SendMessageW.restype = ctypes.c_ssize_t
        hwnd = int(root.frame(), 0)
        # WM_GETICON reads the actual small/title-bar and large/taskbar icon handles.
        original = [user32.SendMessageW(hwnd, 0x007F, size, 0) for size in (0, 1)]
        Inspector(root, 'Branding check', 'TEST')
        shell32 = ctypes.WinDLL('shell32')
        shell32.GetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.POINTER(ctypes.c_wchar_p)]
        shell32.GetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
        app_id = ctypes.c_wchar_p()
        assert shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(app_id)) == 0
        try:
            assert app_id.value == 'SystemX.AIInspector'
        finally:
            ole32 = ctypes.WinDLL('ole32')
            ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
            ole32.CoTaskMemFree(app_id)
        icons = [user32.SendMessageW(hwnd, 0x007F, size, 0) for size in (0, 1)]
        assert all(icons) and all(new != old for new, old in zip(icons, original))
    finally:
        root.destroy()
