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

import ctypes
from collections.abc import Callable
from ctypes import wintypes

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is a sentinel HANDLE value (-4).
_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
_PROCESS_PER_MONITOR_DPI_AWARE = 2  # shcore PROCESS_DPI_AWARENESS


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
