from __future__ import annotations

import subprocess
import xml.sax.saxutils as saxutils
from collections.abc import Callable
from pathlib import Path

from transclip.daemon.common import CommandResult, ServiceState, repo_root, run_command, service_command
from transclip.platform.runtime import PlatformRuntime, get_runtime, user_log_dir
from transclip.product import DISPLAY_NAME, LOG_DIR_NAME, TASK_SCHEDULER_NAME
from transclip.settings import Settings, load_settings

Runner = Callable[..., subprocess.CompletedProcess[str]]


def task_scheduler_xml_path(runtime: PlatformRuntime | None = None) -> Path:
    return user_log_dir(LOG_DIR_NAME, runtime) / f"{TASK_SCHEDULER_NAME}.xml"


def build_task_scheduler_xml(settings_path: Path | None = None) -> str:
    command_parts = service_command(settings_path)
    command = saxutils.escape(command_parts[0])
    arguments = saxutils.escape(subprocess.list2cmdline([str(part) for part in command_parts[1:]]))
    working_directory = saxutils.escape(str(repo_root()))
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-16"?>',
            '<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">',
            "  <RegistrationInfo>",
            f"    <Description>{DISPLAY_NAME} dictation service</Description>",
            "  </RegistrationInfo>",
            "  <Triggers>",
            "    <LogonTrigger>",
            "      <Enabled>true</Enabled>",
            "    </LogonTrigger>",
            "  </Triggers>",
            "  <Principals>",
            '    <Principal id="Author">',
            "      <LogonType>InteractiveToken</LogonType>",
            "      <RunLevel>LeastPrivilege</RunLevel>",
            "    </Principal>",
            "  </Principals>",
            "  <Settings>",
            "    <Enabled>true</Enabled>",
            "    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>",
            "    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>",
            "    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>",
            "    <AllowHardTerminate>true</AllowHardTerminate>",
            "    <StartWhenAvailable>true</StartWhenAvailable>",
            "    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>",
            "    <Hidden>true</Hidden>",
            "    <RestartOnFailure>",
            "      <Interval>PT2M</Interval>",
            "      <Count>3</Count>",
            "    </RestartOnFailure>",
            "  </Settings>",
            '  <Actions Context="Author">',
            "    <Exec>",
            f"      <Command>{command}</Command>",
            f"      <Arguments>{arguments}</Arguments>",
            f"      <WorkingDirectory>{working_directory}</WorkingDirectory>",
            "    </Exec>",
            "  </Actions>",
            "</Task>",
            "",
        ]
    )


def install_windows_daemon(
    settings_path: Path | None = None,
    settings: Settings | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
    *,
    hotkey_setup_message: Callable[..., str],
) -> list[CommandResult]:
    results: list[CommandResult] = []
    platform_runtime = get_runtime(runtime)
    log_root = user_log_dir(LOG_DIR_NAME, platform_runtime)
    log_root.mkdir(parents=True, exist_ok=True)
    xml_path = task_scheduler_xml_path(platform_runtime)
    xml_path.write_text(build_task_scheduler_xml(settings_path), encoding="utf-16")
    results.append(CommandResult(True, f"wrote {xml_path}"))
    results.append(
        run_command(["schtasks", "/Create", "/TN", TASK_SCHEDULER_NAME, "/XML", str(xml_path), "/F"], runner)
    )
    results.append(run_command(["schtasks", "/Run", "/TN", TASK_SCHEDULER_NAME], runner))
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
    results = [
        run_command(["schtasks", "/End", "/TN", TASK_SCHEDULER_NAME], runner, tolerate_failure=True),
        run_command(["schtasks", "/Delete", "/TN", TASK_SCHEDULER_NAME, "/F"], runner, tolerate_failure=True),
    ]
    path = task_scheduler_xml_path(runtime)
    if path.exists():
        path.unlink()
        results.append(CommandResult(True, f"removed {path}"))
    return results


def windows_service_action(action: str, runner: Runner = subprocess.run) -> CommandResult:
    commands = {
        "start": ["schtasks", "/Run", "/TN", TASK_SCHEDULER_NAME],
        "stop": ["schtasks", "/End", "/TN", TASK_SCHEDULER_NAME],
        "restart": ["schtasks", "/End", "/TN", TASK_SCHEDULER_NAME],
    }
    if action == "restart":
        stop = run_command(commands["stop"], runner, tolerate_failure=True)
        start = run_command(["schtasks", "/Run", "/TN", TASK_SCHEDULER_NAME], runner)
        return CommandResult(stop.ok and start.ok, f"{stop.detail}; {start.detail}")
    return run_command(commands[action], runner)


def windows_service_state(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ServiceState:
    path = task_scheduler_xml_path(runtime)
    result = runner(
        ["schtasks", "/Query", "/TN", TASK_SCHEDULER_NAME, "/FO", "LIST", "/V"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    installed = result.returncode == 0 or path.exists()
    active = result.returncode == 0 and _windows_task_reports_running(result.stdout)
    return ServiceState(
        installed=installed,
        active=active,
        detail=result.stdout.strip() or f"exit {result.returncode}",
    )


def _windows_task_reports_running(output: str) -> bool:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("status:"):
            status = stripped.split(":", 1)[1].strip().lower()
            return status == "running"
    return False
