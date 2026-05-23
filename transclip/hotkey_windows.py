from __future__ import annotations

from collections.abc import Callable

from .platform_runtime import PlatformRuntime, get_runtime
from .settings import Settings, active_hotkey


def install_hotkey() -> tuple[bool, str]:
    return True, "Windows hotkey is registered by the tray app when transclip tray is running"


def start_windows_hotkey(
    callback: Callable[[], None],
    settings: Settings,
    runtime: PlatformRuntime | None = None,
) -> Callable[[], None]:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() != "Windows":
        raise RuntimeError("Windows hotkey listener is only available on Windows")
    try:
        import keyboard
    except ImportError as exc:
        raise RuntimeError(
            "keyboard is not installed; install transclip[windows-ui] for global hotkeys"
        ) from exc
    binding = active_hotkey(settings, platform_runtime)
    handle = keyboard.add_hotkey(binding, callback, suppress=False)

    def stop() -> None:
        keyboard.remove_hotkey(handle)

    return stop
