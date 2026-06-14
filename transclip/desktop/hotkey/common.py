from __future__ import annotations

from pathlib import Path

from transclip.platform.runtime import PlatformRuntime
from transclip.settings import Settings, active_hotkey

from .toggle_command import build_toggle_command


def macos_hotkey_setup_message(
    settings: Settings | None = None,
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> str:
    current = settings or Settings()
    binding = current.hotkey_macos
    command = build_toggle_command(settings_path, runtime=runtime)
    return (
        f"Install the native macOS hotkey helper for binding {binding!r}:\n"
        f"transclip install-macos-hotkey\n\n"
        f"Fallback: configure a macOS Keyboard Shortcut or Shortcuts.app action:\n{command}"
    )


def windows_hotkey_setup_message(
    settings: Settings | None = None,
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> str:
    del settings_path
    current = settings or Settings()
    binding = active_hotkey(current, runtime)
    return (
        f"Logon autostart registered. Global hotkey {binding!r} is registered when "
        f"transclip tray is running; change it from the tray Set hotkey menu or hotkey_windows."
    )
