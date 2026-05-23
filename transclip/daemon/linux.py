from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

from transclip.desktop.hotkey import install_shortcut
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.product import DISPLAY_NAME, SERVICE_NAME
from transclip.settings import Settings

from .common import CommandResult, ServiceState, repo_root, run_command, service_command

Runner = Callable[..., subprocess.CompletedProcess[str]]


def systemd_user_unit_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / ".config" / "systemd" / "user" / SERVICE_NAME


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


def uninstall_linux_daemon(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    results = [
        run_command(["systemctl", "--user", "disable", "--now", SERVICE_NAME], runner, tolerate_failure=True)
    ]
    path = systemd_user_unit_path(runtime)
    if path.exists():
        path.unlink()
        results.append(CommandResult(True, f"removed {path}"))
    results.append(run_command(["systemctl", "--user", "daemon-reload"], runner, tolerate_failure=True))
    return results


def linux_service_action(
    action: str,
    runner: Runner = subprocess.run,
) -> CommandResult:
    commands = {
        "start": ["systemctl", "--user", "start", SERVICE_NAME],
        "stop": ["systemctl", "--user", "stop", SERVICE_NAME],
        "restart": ["systemctl", "--user", "restart", SERVICE_NAME],
    }
    return run_command(commands[action], runner)


def linux_service_state(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    platform_runtime = get_runtime(runtime)
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
