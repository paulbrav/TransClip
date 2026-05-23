from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

from transclip.desktop.hotkey import install_shortcut
from transclip.models import normalize_asr_backend
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.product import DISPLAY_NAME, SERVICE_NAME
from transclip.settings import Settings, load_settings

from .common import CommandResult, ServiceState, repo_root, run_command, service_command
from .protocol import PlatformDaemon

Runner = Callable[..., subprocess.CompletedProcess[str]]


def systemd_user_unit_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / ".config" / "systemd" / "user" / SERVICE_NAME


def build_systemd_unit(settings_path: Path | None = None) -> str:
    exec_start = shlex.join(service_command(settings_path))
    settings = load_settings(settings_path)
    service_lines = [
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={repo_root()}",
        f"ExecStart={exec_start}",
        "Restart=on-failure",
        "RestartSec=2",
    ]
    if normalize_asr_backend(settings.asr_backend) == "granite_nar":
        service_lines.extend(
            [
                "Environment=FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE",
                "Environment=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1",
            ]
        )
    return "\n".join(
        [
            "[Unit]",
            f"Description={DISPLAY_NAME} dictation service",
            "After=graphical-session.target",
            "",
            *service_lines,
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
    runtime: PlatformRuntime | None = None,
) -> CommandResult:
    del runtime
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


class LinuxPlatformDaemon:
    def install(
        self,
        *,
        settings_path: Path | None,
        settings: Settings,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]:
        return install_linux_daemon(
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
        return uninstall_linux_daemon(runner=runner, runtime=runtime)

    def service_action(
        self,
        action: str,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> CommandResult:
        return linux_service_action(action, runner=runner, runtime=runtime)

    def service_state(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> ServiceState:
        return linux_service_state(runner=runner, runtime=runtime)


platform_daemon: PlatformDaemon = LinuxPlatformDaemon()
