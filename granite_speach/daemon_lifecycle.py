from __future__ import annotations

import platform
import plistlib
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .gnome_shortcut import (
    GRANITE_SHORTCUT_BINDING,
    build_toggle_command,
    install_gnome_shortcut,
)
from .settings import keywords_path as default_keywords_path
from .settings import write_default_keywords, write_default_settings

Runner = Callable[..., subprocess.CompletedProcess[str]]

SERVICE_NAME = "granite-speach.service"
LAUNCHD_LABEL = "com.paulbrav.granite-speach"


@dataclass(slots=True)
class CommandResult:
    ok: bool
    detail: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def logs_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Logs" / "granite-speach"
    return Path.home() / ".cache" / "granite-speach"


def toggle_log_path() -> Path:
    return logs_dir() / "toggle-record.log"


def systemd_user_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SERVICE_NAME


def launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def service_command(settings_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", "granite_speach.cli"]
    if settings_path:
        command.extend(["--settings", str(settings_path.expanduser().resolve())])
    command.append("serve")
    return command


def build_systemd_unit(settings_path: Path | None = None) -> str:
    exec_start = shlex.join(service_command(settings_path))
    return "\n".join(
        [
            "[Unit]",
            "Description=Granite Speach dictation service",
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


def build_launch_agent(settings_path: Path | None = None) -> bytes:
    log_root = logs_dir()
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": service_command(settings_path),
        "WorkingDirectory": str(repo_root()),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_root / "service.out.log"),
        "StandardErrorPath": str(log_root / "service.err.log"),
        "EnvironmentVariables": {
            "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        },
    }
    return plistlib.dumps(payload, sort_keys=True)


def install_daemon(
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
) -> list[CommandResult]:
    write_default_settings(settings_path)
    keyword_path = default_keywords_path(settings_path.parent) if settings_path else None
    write_default_keywords(keyword_path)
    system = platform.system()
    if system == "Linux":
        return install_linux_daemon(settings_path=settings_path, runner=runner)
    if system == "Darwin":
        return install_macos_daemon(settings_path=settings_path, runner=runner)
    return [CommandResult(False, f"unsupported platform: {system}")]


def install_linux_daemon(
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    unit_path = systemd_user_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(build_systemd_unit(settings_path), encoding="utf-8")
    results.append(CommandResult(True, f"wrote {unit_path}"))
    results.append(run_command(["systemctl", "--user", "daemon-reload"], runner))
    results.append(run_command(["systemctl", "--user", "enable", "--now", SERVICE_NAME], runner))
    try:
        shortcut = install_gnome_shortcut(
            command=build_toggle_command(settings_path),
            binding=GRANITE_SHORTCUT_BINDING,
        )
        results.append(CommandResult(True, f"installed GNOME shortcut {shortcut.binding}: {shortcut.command}"))
    except Exception as exc:
        results.append(CommandResult(False, f"GNOME shortcut install failed: {exc}"))
    return results


def install_macos_daemon(
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    logs_dir().mkdir(parents=True, exist_ok=True)
    plist_path = launch_agent_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(build_launch_agent(settings_path))
    results.append(CommandResult(True, f"wrote {plist_path}"))
    results.append(run_command(["launchctl", "unload", str(plist_path)], runner, tolerate_failure=True))
    results.append(run_command(["launchctl", "load", str(plist_path)], runner))
    results.append(
        CommandResult(
            True,
            "configure a macOS Keyboard Shortcut or Shortcuts.app action to run: "
            + build_toggle_command(settings_path),
        )
    )
    return results


def uninstall_daemon(runner: Runner = subprocess.run) -> list[CommandResult]:
    system = platform.system()
    if system == "Linux":
        results = [
            run_command(["systemctl", "--user", "disable", "--now", SERVICE_NAME], runner, tolerate_failure=True)
        ]
        path = systemd_user_unit_path()
        if path.exists():
            path.unlink()
            results.append(CommandResult(True, f"removed {path}"))
        results.append(run_command(["systemctl", "--user", "daemon-reload"], runner, tolerate_failure=True))
        return results
    if system == "Darwin":
        path = launch_agent_path()
        results = [run_command(["launchctl", "unload", str(path)], runner, tolerate_failure=True)]
        if path.exists():
            path.unlink()
            results.append(CommandResult(True, f"removed {path}"))
        return results
    return [CommandResult(False, f"unsupported platform: {system}")]


def service_action(action: str, runner: Runner = subprocess.run) -> CommandResult:
    system = platform.system()
    if system == "Linux":
        commands = {
            "start": ["systemctl", "--user", "start", SERVICE_NAME],
            "stop": ["systemctl", "--user", "stop", SERVICE_NAME],
            "restart": ["systemctl", "--user", "restart", SERVICE_NAME],
        }
        return run_command(commands[action], runner)
    if system == "Darwin":
        plist_path = str(launch_agent_path())
        commands = {
            "start": ["launchctl", "load", plist_path],
            "stop": ["launchctl", "unload", plist_path],
            "restart": [
                "sh",
                "-lc",
                f"launchctl unload {shlex.quote(plist_path)}; launchctl load {shlex.quote(plist_path)}",
            ],
        }
        return run_command(commands[action], runner)
    return CommandResult(False, f"unsupported platform: {system}")


def service_state(runner: Runner = subprocess.run) -> dict[str, object]:
    system = platform.system()
    if system == "Linux":
        if not shutil.which("systemctl"):
            return {"installed": systemd_user_unit_path().exists(), "active": False, "detail": "systemctl missing"}
        result = runner(
            ["systemctl", "--user", "is-active", SERVICE_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        state = result.stdout.strip() or f"exit {result.returncode}"
        return {
            "installed": systemd_user_unit_path().exists(),
            "active": result.returncode == 0 and state == "active",
            "detail": state,
        }
    if system == "Darwin":
        path = launch_agent_path()
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
