from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from transclip.desktop.hotkey.common import macos_hotkey_setup_message
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.product import LAUNCHD_LABEL
from transclip.settings import load_settings

from .common import CommandResult, ServiceState, logs_dir, repo_root, run_command, service_command

Runner = Callable[..., subprocess.CompletedProcess[str]]


def launch_agent_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def build_launch_agent(
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> bytes:
    import plistlib

    log_root = logs_dir(runtime)
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": service_command(settings_path),
        "WorkingDirectory": str(repo_root()),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_root / "service.out.log"),
        "StandardErrorPath": str(log_root / "service.err.log"),
    }
    return plistlib.dumps(payload, sort_keys=True)


def launchd_gui_domain(runtime: PlatformRuntime | None = None) -> str:
    output = get_runtime(runtime).check_output(["id", "-u"])
    if isinstance(output, bytes):
        output = output.decode()
    uid = output.strip()
    return f"gui/{uid}"


def launchd_target(runtime: PlatformRuntime | None = None) -> str:
    return f"{launchd_gui_domain(runtime)}/{LAUNCHD_LABEL}"


def install_macos_daemon(
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    logs_dir(runtime).mkdir(parents=True, exist_ok=True)
    plist_path = launch_agent_path(runtime)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(build_launch_agent(settings_path, runtime=runtime))
    results.append(CommandResult(True, f"wrote {plist_path}"))
    domain = launchd_gui_domain(runtime)
    target = launchd_target(runtime)
    results.append(run_command(["launchctl", "bootout", target], runner, tolerate_failure=True))
    results.append(run_command(["launchctl", "bootstrap", domain, str(plist_path)], runner))
    results.append(
        CommandResult(
            True,
            macos_hotkey_setup_message(load_settings(settings_path), settings_path, runtime=runtime),
        )
    )
    return results


def uninstall_macos_daemon(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    path = launch_agent_path(runtime)
    target = launchd_target(runtime)
    results = [run_command(["launchctl", "bootout", target], runner, tolerate_failure=True)]
    if path.exists():
        path.unlink()
        results.append(CommandResult(True, f"removed {path}"))
    return results


def macos_service_action(
    action: str,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> CommandResult:
    plist_path = str(launch_agent_path(runtime))
    domain = launchd_gui_domain(runtime)
    target = launchd_target(runtime)
    if action in {"start", "restart"} and _launchd_is_loaded(target, runner):
        return run_command(["launchctl", "kickstart", "-k", target], runner)
    commands = {
        "start": ["launchctl", "bootstrap", domain, plist_path],
        "stop": ["launchctl", "bootout", target],
        "restart": ["launchctl", "bootstrap", domain, plist_path],
    }
    return run_command(commands[action], runner)


def macos_service_state(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    path = launch_agent_path(runtime)
    target = launchd_target(runtime)
    result = runner(
        ["launchctl", "print", target],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    active = result.returncode == 0 and _launchd_print_reports_running(result.stdout)
    return ServiceState(
        installed=path.exists(),
        active=active,
        detail=result.stdout.strip() or f"exit {result.returncode}",
    )


def _launchd_is_loaded(target: str, runner: Runner) -> bool:
    result = runner(
        ["launchctl", "print", target],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _launchd_print_reports_running(output: str) -> bool:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("state ="):
            return stripped.removeprefix("state =").strip() == "running"
        if stripped.startswith("pid ="):
            return True
    return False
