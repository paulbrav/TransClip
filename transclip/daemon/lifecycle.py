from __future__ import annotations

import subprocess
from pathlib import Path

from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.settings import load_settings, write_default_settings

from . import linux as linux_daemon
from . import macos as macos_daemon
from . import windows as windows_daemon
from .common import CommandResult, ServiceState, logs_dir, toggle_log_path
from .protocol import PlatformDaemon, Runner

__all__ = [
    "install_daemon",
    "logs_dir",
    "service_action",
    "service_state",
    "toggle_log_path",
    "uninstall_daemon",
]

_PLATFORM_BACKENDS: dict[str, PlatformDaemon] = {
    "Linux": linux_daemon.platform_daemon,
    "Darwin": macos_daemon.platform_daemon,
    "Windows": windows_daemon.platform_daemon,
}


def _backend(runtime: PlatformRuntime | None = None) -> PlatformDaemon | None:
    return _PLATFORM_BACKENDS.get(get_runtime(runtime).system())


def _unsupported_platform_detail(runtime: PlatformRuntime | None = None) -> str:
    return f"unsupported platform: {get_runtime(runtime).system()}"


def install_daemon(
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    write_default_settings(settings_path)
    settings = load_settings(settings_path)
    backend = _backend(runtime)
    if backend is None:
        return [CommandResult(False, _unsupported_platform_detail(runtime))]
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
        return [CommandResult(False, _unsupported_platform_detail(runtime))]
    return backend.uninstall(runner=runner, runtime=runtime)


def service_action(
    action: str,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> CommandResult:
    backend = _backend(runtime)
    if backend is None:
        return CommandResult(False, _unsupported_platform_detail(runtime))
    return backend.service_action(action, runner=runner, runtime=runtime)


def service_state(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    backend = _backend(runtime)
    if backend is None:
        return ServiceState(
            installed=False,
            active=False,
            detail=_unsupported_platform_detail(runtime),
        )
    return backend.service_state(runner=runner, runtime=runtime)
