from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.settings import Settings, load_settings, write_default_settings

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


class PlatformDaemon(Protocol):
    def install(
        self,
        *,
        settings_path: Path | None,
        settings: Settings,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]: ...

    def uninstall(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]: ...

    def service_action(
        self,
        action: str,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> CommandResult: ...

    def service_state(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> ServiceState: ...


def _daemon_windows():
    from . import windows

    return windows


def _windows_install(
    *,
    settings_path: Path | None,
    settings: Settings,
    runner: Runner,
    runtime: PlatformRuntime | None,
) -> list[CommandResult]:
    from transclip.desktop.hotkey import windows_hotkey_setup_message

    return _daemon_windows().install_windows_daemon(
        settings_path=settings_path,
        settings=settings,
        runner=runner,
        runtime=runtime,
        hotkey_setup_message=windows_hotkey_setup_message,
    )


def _windows_uninstall(*, runner: Runner, runtime: PlatformRuntime | None) -> list[CommandResult]:
    return _daemon_windows().uninstall_windows_daemon(runner=runner, runtime=runtime)


def _linux_service_action(
    action: str,
    *,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> CommandResult:
    return linux_daemon.linux_service_action(action, runner=runner)


def _windows_service_action(
    action: str,
    *,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> CommandResult:
    return _daemon_windows().windows_service_action(action, runner=runner)


def _linux_service_state(
    *,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    return linux_daemon.linux_service_state(runner=runner, runtime=runtime)


def _windows_service_state(
    *,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    return _daemon_windows().windows_service_state(runner=runner, runtime=runtime)


class _PlatformDaemonAdapter:
    def __init__(
        self,
        *,
        install: Callable[..., list[CommandResult]],
        uninstall: Callable[..., list[CommandResult]],
        service_action: Callable[..., CommandResult],
        service_state: Callable[..., ServiceState],
    ) -> None:
        self._install = install
        self._uninstall = uninstall
        self._service_action = service_action
        self._service_state = service_state

    def install(
        self,
        *,
        settings_path: Path | None,
        settings: Settings,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]:
        return self._install(
            settings_path=settings_path,
            settings=settings,
            runner=runner,
            runtime=runtime,
        )

    def uninstall(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]:
        return self._uninstall(runner=runner, runtime=runtime)

    def service_action(
        self,
        action: str,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> CommandResult:
        return self._service_action(action, runner=runner, runtime=runtime)

    def service_state(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> ServiceState:
        return self._service_state(runner=runner, runtime=runtime)


_PLATFORM_BACKENDS: dict[str, PlatformDaemon] = {
    "Linux": _PlatformDaemonAdapter(
        install=linux_daemon.install_linux_daemon,
        uninstall=linux_daemon.uninstall_linux_daemon,
        service_action=_linux_service_action,
        service_state=_linux_service_state,
    ),
    "Darwin": _PlatformDaemonAdapter(
        install=macos_daemon.install_macos_daemon,
        uninstall=macos_daemon.uninstall_macos_daemon,
        service_action=macos_daemon.macos_service_action,
        service_state=macos_daemon.macos_service_state,
    ),
    "Windows": _PlatformDaemonAdapter(
        install=_windows_install,
        uninstall=_windows_uninstall,
        service_action=_windows_service_action,
        service_state=_windows_service_state,
    ),
}


def _backend(runtime: PlatformRuntime | None = None) -> PlatformDaemon | None:
    return _PLATFORM_BACKENDS.get(get_runtime(runtime).system())


def install_daemon(
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    write_default_settings(settings_path)
    settings = load_settings(settings_path)
    backend = _backend(runtime)
    if backend is None:
        system = get_runtime(runtime).system()
        return [CommandResult(False, f"unsupported platform: {system}")]
    return backend.install(
        settings_path=settings_path,
        settings=settings,
        runner=runner,
        runtime=runtime,
    )


def uninstall_daemon(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    backend = _backend(runtime)
    if backend is None:
        system = get_runtime(runtime).system()
        return [CommandResult(False, f"unsupported platform: {system}")]
    return backend.uninstall(runner=runner, runtime=runtime)


def service_action(
    action: str,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> CommandResult:
    backend = _backend(runtime)
    if backend is None:
        system = get_runtime(runtime).system()
        return CommandResult(False, f"unsupported platform: {system}")
    return backend.service_action(action, runner=runner, runtime=runtime)


def service_state(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    backend = _backend(runtime)
    if backend is None:
        system = get_runtime(runtime).system()
        return ServiceState(installed=False, active=False, detail=f"unsupported platform: {system}")
    return backend.service_state(runner=runner, runtime=runtime)
