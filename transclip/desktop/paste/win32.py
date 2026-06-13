from __future__ import annotations

import ctypes
import functools
import platform
import time
from collections.abc import Callable
from ctypes import wintypes

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56

# The clipboard is a single global resource that another process (a clipboard
# manager, browser, Office, RDP) can hold open for a few milliseconds, making
# OpenClipboard fail transiently. Retry a bounded number of times before giving
# up rather than failing the paste outright.
_OPEN_CLIPBOARD_ATTEMPTS = 10
_OPEN_CLIPBOARD_RETRY_DELAY_S = 0.015


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulonglong),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulonglong),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT(ctypes.Structure):
    class _INPUTUNION(ctypes.Union):
        # SendInput validates cbSize against the real Win32 INPUT (40 bytes on
        # x64). MOUSEINPUT is the largest union member, so it must be present
        # for sizeof(INPUT) to match even though we only ever send keystrokes;
        # omitting it makes the struct 32 bytes and SendInput rejects the call.
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUTUNION),
    ]


def _require_windows() -> None:
    if platform.system() != "Windows":
        raise RuntimeError("Win32 clipboard APIs are only available on Windows")


def _configure_prototypes(user32: ctypes.WinDLL, kernel32: ctypes.WinDLL) -> None:
    """Declare 64-bit-safe argument and return types for the Win32 calls.

    Without an explicit ``restype``, ctypes assumes a 32-bit ``int`` return and
    truncates the 64-bit ``HANDLE``/pointer values these functions return. The
    truncated pointer then faults when dereferenced (``GlobalLock`` ->
    ``wstring_at``).
    """
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL


@functools.cache
def _win32_libraries() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    """Bind private ``user32``/``kernel32`` handles with configured prototypes.

    A private ``WinDLL`` keeps our argtypes/restype declarations off the
    process-global ``ctypes.windll`` cache, whose function objects are shared
    with any other code in the process that imports them.
    """
    user32 = ctypes.WinDLL("user32")
    kernel32 = ctypes.WinDLL("kernel32")
    _configure_prototypes(user32, kernel32)
    return user32, kernel32


def _open_clipboard(
    user32: ctypes.WinDLL,
    *,
    attempts: int = _OPEN_CLIPBOARD_ATTEMPTS,
    sleep: Callable[[float], object] = time.sleep,
) -> None:
    """Open the clipboard, retrying transient contention before failing."""
    for attempt in range(attempts):
        if user32.OpenClipboard(None):
            return
        if attempt + 1 < attempts:
            sleep(_OPEN_CLIPBOARD_RETRY_DELAY_S)
    raise RuntimeError(f"OpenClipboard failed after {attempts} attempts")


def read_clipboard_text() -> str:
    _require_windows()
    user32, kernel32 = _win32_libraries()
    _open_clipboard(user32)
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        locked = kernel32.GlobalLock(handle)
        if not locked:
            raise RuntimeError("GlobalLock failed")
        try:
            return ctypes.wstring_at(locked)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def write_clipboard_text(text: str) -> None:
    _require_windows()
    user32, kernel32 = _win32_libraries()
    _open_clipboard(user32)
    try:
        if not user32.EmptyClipboard():
            raise RuntimeError("EmptyClipboard failed")
        payload = (text + "\0").encode("utf-16-le")
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(payload))
        if not handle:
            raise RuntimeError("GlobalAlloc failed")
        locked = kernel32.GlobalLock(handle)
        if not locked:
            kernel32.GlobalFree(handle)
            raise RuntimeError("GlobalLock failed")
        ctypes.memmove(locked, payload, len(payload))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            raise RuntimeError("SetClipboardData failed")
    finally:
        user32.CloseClipboard()


def _keyboard_input(vk: int, *, key_up: bool = False) -> INPUT:
    event = INPUT()
    event.type = INPUT_KEYBOARD
    event.ki.wVk = vk
    event.ki.dwFlags = KEYEVENTF_KEYUP if key_up else 0
    return event


def send_ctrl_v_paste() -> None:
    _require_windows()
    user32, _kernel32 = _win32_libraries()
    inputs = (
        _keyboard_input(VK_CONTROL),
        _keyboard_input(VK_V),
        _keyboard_input(VK_V, key_up=True),
        _keyboard_input(VK_CONTROL, key_up=True),
    )
    array = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise RuntimeError(f"SendInput returned {sent}, expected {len(inputs)}")
    time.sleep(0.05)
