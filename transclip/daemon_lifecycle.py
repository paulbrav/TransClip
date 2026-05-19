from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .gnome_shortcut import (
    build_toggle_command,
    install_shortcut,
)
from .platform_runtime import PlatformRuntime, get_runtime, user_log_dir
from .product import DISPLAY_NAME, IMPORT_PACKAGE, LAUNCHD_LABEL, LOG_DIR_NAME, SERVICE_NAME
from .settings import Settings, default_config_dir, load_settings, write_default_settings

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(slots=True)
class CommandResult:
    ok: bool
    detail: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def service_working_directory(runtime: PlatformRuntime | None = None) -> Path:
    del runtime
    return default_config_dir()


def service_environment_variables(system: str) -> dict[str, str]:
    if system != "Linux":
        return {}
    return {
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
    }


def launchctl_domain() -> str:
    return f"gui/{os.getuid()}"


def logs_dir(runtime: PlatformRuntime | None = None) -> Path:
    return user_log_dir(LOG_DIR_NAME, runtime)


def toggle_log_path(runtime: PlatformRuntime | None = None) -> Path:
    return logs_dir(runtime) / "toggle-record.log"


def systemd_user_unit_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / ".config" / "systemd" / "user" / SERVICE_NAME


def launch_agent_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def service_command(settings_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", f"{IMPORT_PACKAGE}.cli"]
    if settings_path:
        command.extend(["--settings", str(settings_path.expanduser().resolve())])
    command.append("serve")
    return command


def build_systemd_unit(settings_path: Path | None = None) -> str:
    exec_start = shlex.join(service_command(settings_path))
    lines = [
        "[Unit]",
        f"Description={DISPLAY_NAME} dictation service",
        "After=graphical-session.target",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={service_working_directory()}",
        f"ExecStart={exec_start}",
        "Restart=on-failure",
        "RestartSec=2",
    ]
    for key, value in service_environment_variables("Linux").items():
        lines.append(f"Environment={key}={value}")
    lines.extend(
        [
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )
    return "\n".join(lines)


def build_launch_agent(
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> bytes:
    platform_runtime = get_runtime(runtime)
    log_root = logs_dir(platform_runtime)
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": service_command(settings_path),
        "WorkingDirectory": str(service_working_directory(platform_runtime)),
        "RunAtLoad": True,
        "KeepAlive": True,
        "LimitLoadToSessionType": "Aqua",
        "StandardOutPath": str(log_root / "service.out.log"),
        "StandardErrorPath": str(log_root / "service.err.log"),
        "EnvironmentVariables": service_environment_variables(platform_runtime.system()),
    }
    return plistlib.dumps(payload, sort_keys=True)


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
        return install_macos_daemon(
            settings_path=settings_path,
            settings=settings,
            runner=runner,
            runtime=runtime,
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
    settings: Settings | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    settings = settings or load_settings(settings_path)
    results: list[CommandResult] = []
    logs_dir(runtime).mkdir(parents=True, exist_ok=True)
    plist_path = launch_agent_path(runtime)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(build_launch_agent(settings_path, runtime=runtime))
    results.append(CommandResult(True, f"wrote {plist_path}"))
    domain = launchctl_domain()
    results.append(run_command(["launchctl", "bootout", domain, str(plist_path)], runner, tolerate_failure=True))
    results.append(run_command(["launchctl", "bootstrap", domain, str(plist_path)], runner))
    results.append(
        CommandResult(
            True,
            "configure a macOS Keyboard Shortcut or Shortcuts.app action "
            f"(suggested binding {settings.hotkey_macos}) to run: "
            + build_toggle_command(settings_path, runtime=runtime),
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
        domain = launchctl_domain()
        results = [run_command(["launchctl", "bootout", domain, str(path)], runner, tolerate_failure=True)]
        if path.exists():
            path.unlink()
            results.append(CommandResult(True, f"removed {path}"))
        return results
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
        domain = launchctl_domain()
        commands = {
            "start": ["launchctl", "bootstrap", domain, plist_path],
            "stop": ["launchctl", "bootout", domain, plist_path],
            "restart": [
                "sh",
                "-lc",
                "launchctl bootout "
                + shlex.quote(domain)
                + " "
                + shlex.quote(plist_path)
                + "; launchctl bootstrap "
                + shlex.quote(domain)
                + " "
                + shlex.quote(plist_path),
            ],
        }
        return run_command(commands[action], runner)
    return CommandResult(False, f"unsupported platform: {system}")


def service_state(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> dict[str, object]:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system == "Linux":
        if not platform_runtime.which("systemctl"):
            return {
                "installed": systemd_user_unit_path(runtime).exists(),
                "active": False,
                "detail": "systemctl missing",
            }
        result = runner(
            ["systemctl", "--user", "is-active", SERVICE_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        state = result.stdout.strip() or f"exit {result.returncode}"
        return {
            "installed": systemd_user_unit_path(runtime).exists(),
            "active": result.returncode == 0 and state == "active",
            "detail": state,
        }
    if system == "Darwin":
        path = launch_agent_path(runtime)
        result = runner(
            ["launchctl", "list", LAUNCHD_LABEL],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return {
            "installed": path.exists(),
            "active": result.returncode == 0,
            "detail": result.stdout.strip() or f"exit {result.returncode}",
        }
    return {"installed": False, "active": False, "detail": f"unsupported platform: {system}"}


def run_command(
    command: list[str],
    runner: Runner,
    tolerate_failure: bool = False,
) -> CommandResult:
    try:
        result = runner(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(tolerate_failure, f"{command[0]} missing: {exc}")
    output = result.stdout.strip()
    ok = result.returncode == 0 or tolerate_failure
    detail = shlex.join(command)
    if output:
        detail += f": {output}"
    elif result.returncode != 0:
        detail += f": exit {result.returncode}"
    return CommandResult(ok, detail)
