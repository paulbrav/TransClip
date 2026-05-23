from __future__ import annotations

from pathlib import Path

from transclip.platform.runtime import PlatformRuntime, get_runtime

from . import common, toggle_command, windows

__all__ = [
    "build_toggle_command",
    "build_toggle_invocation",
    "get_gnome_shortcut_status",
    "hotkey_setup_message",
    "install_shortcut",
    "macos_hotkey_setup_message",
    "shortcut_readiness",
    "start_windows_hotkey",
    "toggle_log_shell_path",
    "windows_hotkey_setup_message",
]

build_toggle_command = toggle_command.build_toggle_command
build_toggle_invocation = toggle_command.build_toggle_invocation
toggle_log_shell_path = toggle_command.toggle_log_shell_path
macos_hotkey_setup_message = common.macos_hotkey_setup_message
windows_hotkey_setup_message = common.windows_hotkey_setup_message


def install_shortcut(
    settings_path: Path | None = None,
    binding: str | None = None,
    runtime: PlatformRuntime | None = None,
):
    if get_runtime(runtime).system() != "Linux":
        raise RuntimeError("shortcut install is only supported on Linux GNOME")
    from . import linux_gnome

    return linux_gnome.install_shortcut(
        settings_path=settings_path,
        binding=binding,
        runtime=runtime,
    )


def get_gnome_shortcut_status(*args, **kwargs):
    from . import linux_gnome

    return linux_gnome.get_gnome_shortcut_status(*args, **kwargs)


def shortcut_readiness(*args, **kwargs):
    from . import linux_gnome

    return linux_gnome.shortcut_readiness(*args, **kwargs)


def hotkey_setup_message(
    settings=None,
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> str:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system == "Darwin":
        return common.macos_hotkey_setup_message(settings, settings_path, runtime=runtime)
    if system == "Windows":
        return common.windows_hotkey_setup_message(settings, settings_path, runtime=runtime)
    return "Hotkey setup is handled by the platform shortcut installer."


def start_windows_hotkey(*args, **kwargs):
    return windows.start_windows_hotkey(*args, **kwargs)
