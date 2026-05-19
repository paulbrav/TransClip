from __future__ import annotations

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
from .settings import Settings, load_settings, write_default_settings

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(slots=True)
class CommandResult:
    ok: bool
    detail: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
            "configure a macOS Keyboard Shortcut or Shortcuts.app action to run: "
            + build_toggle_command(settings_path),
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
        if action == "start" and _launchd_is_loaded(target, runner):
            return run_command(["launchctl", "kickstart", "-k", target], runner)
        commands = {
            "start": ["launchctl", "bootstrap", domain, plist_path],
            "stop": ["launchctl", "bootout", target],
            "restart": ["launchctl", "kickstart", "-k", target],
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
        target = launchd_target(runtime)
        result = runner(
            ["launchctl", "print", target],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        active = result.returncode == 0 and _launchd_print_reports_running(result.stdout)
        return {
            "installed": path.exists(),
            "active": active,
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
