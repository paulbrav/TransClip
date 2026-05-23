from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from transclip.paths import service_settings_path
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.product import IMPORT_PACKAGE


def build_toggle_invocation(settings_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", f"{IMPORT_PACKAGE}.cli"]
    if settings_path:
        command.extend(["--settings", service_settings_path(settings_path)])
    command.extend(["toggle-record", "--paste"])
    return command


def toggle_log_shell_path(runtime: PlatformRuntime | None = None) -> str:
    from transclip.daemon.common import toggle_log_path

    return str(toggle_log_path(runtime))


def build_toggle_command(
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> str:
    platform_runtime = get_runtime(runtime)
    command = build_toggle_invocation(settings_path)
    log_path = toggle_log_shell_path(runtime)
    if platform_runtime.system() == "Windows":
        quoted_log_path = log_path.replace("'", "''")
        invocation = subprocess.list2cmdline([str(part) for part in command])
        ps_command = (
            f"New-Item -ItemType Directory -Force -Path (Split-Path -LiteralPath '{quoted_log_path}') "
            f"| Out-Null; {invocation} >> '{quoted_log_path}' 2>&1"
        )
        return subprocess.list2cmdline(["powershell", "-NoProfile", "-Command", ps_command])
    quoted_log_path = shlex.quote(log_path)
    script = f'mkdir -p "$(dirname {quoted_log_path})"; ' + shlex.join(command) + f" >> {quoted_log_path} 2>&1"
    return shlex.join(["/bin/sh", "-lc", script])
