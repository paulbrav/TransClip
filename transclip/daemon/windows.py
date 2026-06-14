from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from transclip.daemon.common import (
    CommandResult,
    ServiceState,
    repo_root,
    run_command,
    service_command,
)
from transclip.daemon.protocol import PlatformDaemon
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.product import DISPLAY_NAME
from transclip.settings import Settings, load_settings

Runner = Callable[..., subprocess.CompletedProcess[str]]

# Per-user autostart. The logon shell (explorer.exe) launches every value under
# this key at interactive sign-in. It is writable by the user's own token, so -
# unlike a root-folder Task Scheduler task - registering autostart here needs no
# elevation. That was the whole reason for the switch: ``schtasks /Create``
# failed with "Access is denied" for a non-admin user.
RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_KEY_VALUE_NAME = DISPLAY_NAME

# Pre-Run-key TransClip versions registered a root-folder Task Scheduler logon
# task with this name. We migrated to the HKCU Run key above; install/uninstall
# both clear any leftover task so an in-place upgrade does not double-start the
# service at logon (old task + Run key) or orphan the task on uninstall.
_LEGACY_TASK_NAME = "TransClip"


# --- registry seams -------------------------------------------------------
# ``winreg`` is a Windows-only stdlib module, but this file is imported on every
# platform (CI runs the daemon tests on Linux). Import it lazily inside each
# helper so the module loads everywhere; cross-platform tests patch these three
# functions, and a win32-gated test exercises the real round-trip.


def _set_autostart(command: str) -> None:
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, RUN_KEY_VALUE_NAME, 0, winreg.REG_SZ, command)


def _autostart_command() -> str | None:
    import winreg

    try:
        with winreg.OpenKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_QUERY_VALUE) as key:
            value, _kind = winreg.QueryValueEx(key, RUN_KEY_VALUE_NAME)
    except FileNotFoundError:
        # The Run key always exists, but the value (or, defensively, the key) may
        # not - QueryValueEx raises FileNotFoundError when the value is absent.
        return None
    return str(value)


def _clear_autostart() -> bool:
    import winreg

    try:
        with winreg.OpenKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, RUN_KEY_VALUE_NAME)
    except FileNotFoundError:
        return False
    return True


# --- process control ------------------------------------------------------


def _spawn_service(command: str) -> None:
    """Launch the dictation service detached from this console.

    ``DETACHED_PROCESS`` + ``CREATE_NEW_PROCESS_GROUP`` cut the child loose so it
    outlives the launching ``transclip install``/``start`` process, and
    ``CREATE_NO_WINDOW`` suppresses the console flash (belt-and-suspenders with
    the ``pythonw.exe`` GUI-subsystem interpreter baked into ``service_command``).
    """
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        )
    subprocess.Popen(command, cwd=str(repo_root()), close_fds=True, creationflags=creationflags)


def _start_now(command: str) -> CommandResult:
    try:
        _spawn_service(command)
    except OSError as exc:
        return CommandResult(False, f"could not start dictation service: {exc}")
    return CommandResult(True, "started dictation service")


def _service_command_line(settings_path: Path | None = None) -> str:
    return subprocess.list2cmdline(service_command(settings_path))


# PowerShell matches the live service by command line: the interpreter is
# ``pythonw.exe`` (or ``python.exe``), so the distinguishing marker is
# ``transclip ... serve``. ``serve`` keeps this from catching the tray
# (``transclip tray``) or a toggle invocation (``transclip ... toggle``).
#
# The image-name guard (``$_.Name -like 'python*'``) is load-bearing, not an
# optimisation: ``Get-CimInstance`` enumerates *this very PowerShell process*,
# whose own command line contains the literals ``*transclip*`` and ``*serve*``
# (they are in the ``-Command`` text). Without the guard the query counts and
# tries to kill itself - a phantom "1 service process running" after uninstall,
# and a "cannot find process" error when that PID exits mid-pipeline.
_SERVICE_PROCESS_FILTER = (
    "Get-CimInstance Win32_Process | Where-Object { "
    "$_.Name -like 'python*' -and $_.CommandLine -like '*transclip*' -and $_.CommandLine -like '*serve*' }"
)


def _count_service_processes(runner: Runner) -> int:
    result = runner(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"@({_SERVICE_PROCESS_FILTER}).Count"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0
    text = (result.stdout or "").strip()
    return int(text) if text.isdigit() else 0


def _stop_service(runner: Runner) -> CommandResult:
    return run_command(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"{_SERVICE_PROCESS_FILTER} | ForEach-Object {{ "
            "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
        ],
        runner,
        tolerate_failure=True,
    )


def _remove_legacy_task(runner: Runner) -> CommandResult:
    """Best-effort teardown of the pre-Run-key Task Scheduler logon task.

    No-ops cleanly when no such task exists (schtasks returns nonzero, tolerated).
    """
    run_command(["schtasks", "/End", "/TN", _LEGACY_TASK_NAME], runner, tolerate_failure=True)
    run_command(["schtasks", "/Delete", "/TN", _LEGACY_TASK_NAME, "/F"], runner, tolerate_failure=True)
    return CommandResult(True, f"cleared any legacy Task Scheduler task ({_LEGACY_TASK_NAME})")


# --- public API -----------------------------------------------------------


def install_windows_daemon(
    settings_path: Path | None = None,
    settings: Settings | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
    *,
    hotkey_setup_message: Callable[..., str],
) -> list[CommandResult]:
    platform_runtime = get_runtime(runtime)
    command = _service_command_line(settings_path)
    _set_autostart(command)
    results: list[CommandResult] = [
        _remove_legacy_task(runner),
        CommandResult(True, f"registered logon autostart ({RUN_KEY_VALUE_NAME}): {command}"),
        _start_now(command),
    ]
    settings = settings or load_settings(settings_path, runtime=platform_runtime)
    results.append(
        CommandResult(
            True,
            hotkey_setup_message(settings, settings_path, runtime=platform_runtime),
        )
    )
    if settings.asr_model:
        results.append(
            CommandResult(
                True,
                f"prefetch Granite AR model: transclip models prefetch --model {settings.asr_model}",
            )
        )
    return results


def uninstall_windows_daemon(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    del runtime
    results = [_stop_service(runner), _remove_legacy_task(runner)]
    if _clear_autostart():
        results.append(CommandResult(True, f"removed logon autostart ({RUN_KEY_VALUE_NAME})"))
    else:
        results.append(CommandResult(True, "logon autostart was not registered"))
    return results


def windows_service_action(
    action: str,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> CommandResult:
    del runtime
    if action == "start":
        return _start_now(_autostart_command() or _service_command_line())
    if action == "stop":
        return _stop_service(runner)
    if action == "restart":
        stop = _stop_service(runner)
        start = _start_now(_autostart_command() or _service_command_line())
        return CommandResult(start.ok, f"{stop.detail}; {start.detail}")
    raise ValueError(f"unknown service action: {action}")


def windows_service_state(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    del runtime
    command = _autostart_command()
    installed = command is not None
    running = _count_service_processes(runner)
    active = running > 0
    if installed and active:
        detail = f"autostart registered; {running} service process(es) running"
    elif installed:
        detail = "autostart registered; service not running"
    elif active:
        detail = f"{running} service process(es) running; autostart not registered"
    else:
        detail = "autostart not registered; service not running"
    return ServiceState(installed=installed, active=active, detail=detail)


class WindowsPlatformDaemon:
    def install(
        self,
        *,
        settings_path: Path | None,
        settings: Settings,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]:
        from transclip.desktop.hotkey import windows_hotkey_setup_message

        return install_windows_daemon(
            settings_path=settings_path,
            settings=settings,
            runner=runner,
            runtime=runtime,
            hotkey_setup_message=windows_hotkey_setup_message,
        )

    def uninstall(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]:
        return uninstall_windows_daemon(runner=runner, runtime=runtime)

    def service_action(
        self,
        action: str,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> CommandResult:
        return windows_service_action(action, runner=runner, runtime=runtime)

    def service_state(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> ServiceState:
        return windows_service_state(runner=runner, runtime=runtime)


platform_daemon: PlatformDaemon = WindowsPlatformDaemon()
