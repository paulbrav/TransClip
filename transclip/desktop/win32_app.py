"""Windows process-setup helpers for the tray/GUI process.

These configure process-wide Windows attributes that should be set once, early,
when the interactive (tray) process starts: DPI awareness so windows render
crisply, the AppUserModelID so the shell attributes the app correctly, and a
single-instance guard so a second tray cannot double-register the hotkey.

Each real syscall degrades to a no-op off Windows (``ctypes.windll`` does not
exist there), so the module imports everywhere and the helpers are safe to call
unconditionally.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
from collections.abc import Callable
from ctypes import wintypes

from transclip.product import APP_USER_MODEL_ID

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is a sentinel HANDLE value (-4).
_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
_PROCESS_PER_MONITOR_DPI_AWARE = 2  # shcore PROCESS_DPI_AWARENESS
_ERROR_ALREADY_EXISTS = 183

# Per-session (Local namespace) mutex name; one interactive tray per login.
SINGLE_INSTANCE_MUTEX = "TransClip-tray-singleton"


@functools.cache
def _kernel32() -> ctypes.WinDLL:
    """Private kernel32 handle with use_last_error so get_last_error is reliable.

    Raises AttributeError off Windows (ctypes.WinDLL does not exist there), which
    the callers catch and degrade to a no-op.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _set_dpi_context() -> bool:
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDpiAwarenessContext.argtypes = [wintypes.HANDLE]
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        return bool(user32.SetProcessDpiAwarenessContext(_PER_MONITOR_AWARE_V2))
    except (AttributeError, OSError):
        return False


def _set_dpi_shcore() -> bool:
    try:
        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long  # HRESULT
        return shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE) == 0  # S_OK
    except (AttributeError, OSError):
        return False


def _set_dpi_legacy() -> bool:
    try:
        return bool(ctypes.windll.user32.SetProcessDPIAware())
    except (AttributeError, OSError):
        return False


def set_dpi_awareness(
    set_context: Callable[[], bool] = _set_dpi_context,
    set_shcore: Callable[[], bool] = _set_dpi_shcore,
    set_legacy: Callable[[], bool] = _set_dpi_legacy,
) -> str:
    """Opt the process into the best available DPI awareness mode.

    Microsoft recommends Per-Monitor-v2 (Win10 1703+) so windows render crisply
    instead of being bitmap-stretched (blurry) on high-DPI / multi-monitor
    setups. Falls back to per-monitor (shcore, Win8.1) then system awareness on
    older Windows. Returns the mode that was applied.
    """
    if set_context():
        return "per-monitor-v2"
    if set_shcore():
        return "per-monitor"
    if set_legacy():
        return "system"
    return "unaware"


def set_app_user_model_id(aumid: str = APP_USER_MODEL_ID) -> bool:
    """Give the process an explicit AppUserModelID for the Windows shell.

    Without one, the shell derives an implicit identity from the host
    executable (python.exe/pythonw.exe), which groups TransClip under Python on
    the taskbar and misattributes notifications. Must be called before any
    window is created. Returns True on success (S_OK).
    """
    try:
        shell32 = ctypes.windll.shell32
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [wintypes.LPCWSTR]
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long  # HRESULT
        return shell32.SetCurrentProcessExplicitAppUserModelID(aumid) == 0
    except (AttributeError, OSError):
        return False


def acquire_single_instance(name: str = SINGLE_INSTANCE_MUTEX) -> int | None:
    """Acquire a named mutex; return its handle, or None if already held.

    A second ``transclip tray`` would otherwise install a second low-level
    keyboard hook and make the toggle hotkey fire twice. The handle must be kept
    alive for the process lifetime (the mutex releases when it is closed or the
    process exits). Returns None off Windows, where there is nothing to guard.
    """
    try:
        kernel32 = _kernel32()
    except (AttributeError, OSError):
        return None
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        return None
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return handle


def release_single_instance(handle: int | None) -> None:
    if not handle:
        return
    with contextlib.suppress(AttributeError, OSError):
        _kernel32().CloseHandle(handle)
