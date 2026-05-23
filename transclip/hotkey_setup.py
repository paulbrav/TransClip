from __future__ import annotations

import shlex
import sys
from pathlib import Path

from .platform_runtime import PlatformRuntime, user_log_dir
from .product import IMPORT_PACKAGE, LOG_DIR_NAME
from .settings import Settings


def build_toggle_invocation(settings_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", f"{IMPORT_PACKAGE}.cli"]
    if settings_path:
        command.extend(["--settings", str(settings_path.expanduser().resolve())])
    command.extend(["toggle-record", "--paste"])
    return command


def toggle_log_shell_path(runtime: PlatformRuntime | None = None) -> str:
    log_dir = user_log_dir(LOG_DIR_NAME, runtime)
    return str(log_dir / "toggle-record.log")


def build_toggle_command(
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> str:
    command = build_toggle_invocation(settings_path)
    log_path = toggle_log_shell_path(runtime)
    quoted_log_path = shlex.quote(log_path)
    script = f'mkdir -p "$(dirname {quoted_log_path})"; ' + shlex.join(command) + f" >> {quoted_log_path} 2>&1"
    return shlex.join(["/bin/sh", "-lc", script])


def macos_hotkey_setup_message(
    settings: Settings | None = None,
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> str:
    current = settings or Settings()
    binding = current.hotkey_macos
    command = build_toggle_command(settings_path, runtime=runtime)
    return (
        f"Configure a macOS Keyboard Shortcut or Shortcuts.app action for binding {binding!r}:\n"
        f"{command}"
    )
