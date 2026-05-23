from __future__ import annotations

import plistlib
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

from .daemon_common import CommandResult, ServiceState, repo_root, run_command, service_command
from .gnome_shortcut import install_shortcut
from .hotkey_setup import macos_hotkey_setup_message, windows_hotkey_setup_message
from .platform_runtime import PlatformRuntime, get_runtime, user_log_dir
from .product import DISPLAY_NAME, LAUNCHD_LABEL, LOG_DIR_NAME, SERVICE_NAME
from .settings import Settings, load_settings, write_default_settings

Runner = Callable[..., subprocess.CompletedProcess[str]]

__all__ = [
    "CommandResult",
    "ServiceState",
    "build_systemd_unit",
    "install_daemon",
    "install_linux_daemon",
    "install_macos_daemon",
    "run_command",
    "service_action",
    "service_command",
    "service_state",
    "toggle_log_path",
    "uninstall_daemon",
]


def _daemon_windows():
    from . import daemon_windows

    return daemon_windows


def logs_dir(runtime: PlatformRuntime | None = None) -> Path:
    return user_log_dir(LOG_DIR_NAME, runtime)


def toggle_log_path(runtime: PlatformRuntime | None = None) -> Path:
    return logs_dir(runtime) / "toggle-record.log"


def systemd_user_unit_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / ".config" / "systemd" / "user" / SERVICE_NAME


def launch_agent_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def build_systemd_unit(settings_path: Path | None = None) -> str:
    exec_start = shlex.join(service_command(settings_path))
    return "\n".join(
        [
            "[Unit]",
            f"Description={DISPLAY_NAME} dictation service",
            "After=graphical-session.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={repo_root()}",
            f"ExecStart={exec_start}",
            "Restart=on-failure",
            "RestartSec=2",
            "Environment=FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE",
            "Environment=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def build_launch_agent(
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> bytes:
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


def install_daemon(
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    write_default_settings(settings_path)
    settings = load_settings(settings_path)
    system = get_runtime(runtime).system()
    if system == "Linux":
        return install_linux_daemon(settings_path=settings_path, settings=settings, runner=runner, runtime=runtime)
    if system == "Darwin":
        return install_macos_daemon(settings_path=settings_path, runner=runner, runtime=runtime)
    if system == "Windows":
        return _daemon_windows().install_windows_daemon(
            settings_path=settings_path,
            settings=settings,
            runner=runner,
            runtime=runtime,
            hotkey_setup_message=windows_hotkey_setup_message,
        )
    return [CommandResult(False, f"unsupported platform: {system}")]


def install_linux_daemon(
    settings_path: Path | None = None,
    settings: Settings | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    unit_path = systemd_user_unit_path(runtime)
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(build_systemd_unit(settings_path), encoding="utf-8")
    results.append(CommandResult(True, f"wrote {unit_path}"))
    results.append(run_command(["systemctl", "--user", "daemon-reload"], runner))
    results.append(run_command(["systemctl", "--user", "enable", "--now", SERVICE_NAME], runner))
    try:
        settings = settings or Settings()
        shortcut = install_shortcut(
            settings_path=settings_path,
            binding=settings.hotkey_linux,
        )
        results.append(CommandResult(True, f"installed GNOME shortcut {shortcut.binding}: {shortcut.command}"))
    except Exception as exc:
        results.append(CommandResult(False, f"GNOME shortcut install failed: {exc}"))
    return results


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


def uninstall_daemon(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    system = get_runtime(runtime).system()
    if system == "Linux":
        results = [
            run_command(["systemctl", "--user", "disable", "--now", SERVICE_NAME], runner, tolerate_failure=True)
        ]
        path = systemd_user_unit_path(runtime)
        if path.exists():
            path.unlink()
            results.append(CommandResult(True, f"removed {path}"))
        results.append(run_command(["systemctl", "--user", "daemon-reload"], runner, tolerate_failure=True))
        return results
    if system == "Darwin":
        path = launch_agent_path(runtime)
        target = launchd_target(runtime)
        results = [run_command(["launchctl", "bootout", target], runner, tolerate_failure=True)]
        if path.exists():
            path.unlink()
            results.append(CommandResult(True, f"removed {path}"))
        return results
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
        commands = {
            "start": ["systemctl", "--user", "start", SERVICE_NAME],
            "stop": ["systemctl", "--user", "stop", SERVICE_NAME],
            "restart": ["systemctl", "--user", "restart", SERVICE_NAME],
        }
        return run_command(commands[action], runner)
    if system == "Darwin":
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
        if not platform_runtime.which("systemctl"):
            return ServiceState(
                installed=systemd_user_unit_path(runtime).exists(),
                active=False,
                detail="systemctl missing",
            )
        result = runner(
            ["systemctl", "--user", "is-active", SERVICE_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        state = result.stdout.strip() or f"exit {result.returncode}"
        return ServiceState(
            installed=systemd_user_unit_path(runtime).exists(),
            active=result.returncode == 0 and state == "active",
            detail=state,
        )
    if system == "Darwin":
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
    if system == "Windows":
        return _daemon_windows().windows_service_state(runner=runner, runtime=runtime)
    return ServiceState(installed=False, active=False, detail=f"unsupported platform: {system}")


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
