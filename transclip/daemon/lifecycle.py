from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from transclip.desktop.hotkey.common import windows_hotkey_setup_message
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.settings import load_settings, write_default_settings

from . import linux as linux_daemon
from . import macos as macos_daemon
from .common import CommandResult, ServiceState, logs_dir, toggle_log_path

Runner = Callable[..., subprocess.CompletedProcess[str]]

__all__ = [
    "install_daemon",
    "logs_dir",
    "service_action",
    "service_state",
    "toggle_log_path",
    "uninstall_daemon",
]


def _daemon_windows():
    from . import windows

    return windows


def install_daemon(
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    write_default_settings(settings_path)
    settings = load_settings(settings_path)
    system = get_runtime(runtime).system()
    if system == "Linux":
        return linux_daemon.install_linux_daemon(
            settings_path=settings_path,
            settings=settings,
            runner=runner,
            runtime=runtime,
        )
    if system == "Darwin":
        return macos_daemon.install_macos_daemon(
            settings_path=settings_path,
            runner=runner,
            runtime=runtime,
        )
    if system == "Windows":
        return _daemon_windows().install_windows_daemon(
            settings_path=settings_path,
            settings=settings,
            runner=runner,
            runtime=runtime,
            hotkey_setup_message=windows_hotkey_setup_message,
        )
    return [CommandResult(False, f"unsupported platform: {system}")]


def uninstall_daemon(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    system = get_runtime(runtime).system()
    if system == "Linux":
        return linux_daemon.uninstall_linux_daemon(runner=runner, runtime=runtime)
    if system == "Darwin":
        return macos_daemon.uninstall_macos_daemon(runner=runner, runtime=runtime)
    if system == "Windows":
        return _daemon_windows().uninstall_windows_daemon(runner=runner, runtime=runtime)
    return [CommandResult(False, f"unsupported platform: {system}")]


def service_action(
    action: str,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> CommandResult:
    system = get_runtime(runtime).system()
    if system == "Linux":
        return linux_daemon.linux_service_action(action, runner=runner)
    if system == "Darwin":
        return macos_daemon.macos_service_action(action, runner=runner, runtime=runtime)
    if system == "Windows":
        return _daemon_windows().windows_service_action(action, runner=runner)
    return CommandResult(False, f"unsupported platform: {system}")


def service_state(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system == "Linux":
        return linux_daemon.linux_service_state(runner=runner, runtime=runtime)
    if system == "Darwin":
        return macos_daemon.macos_service_state(runner=runner, runtime=runtime)
    if system == "Windows":
        return _daemon_windows().windows_service_state(runner=runner, runtime=runtime)
    return ServiceState(installed=False, active=False, detail=f"unsupported platform: {system}")
