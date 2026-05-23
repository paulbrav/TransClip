from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .product import IMPORT_PACKAGE

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(slots=True)
class CommandResult:
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ServiceState:
    installed: bool
    active: bool
    detail: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def service_settings_path(settings_path: Path) -> str:
    expanded = settings_path.expanduser()
    text = str(expanded)
    if expanded.is_absolute() or (len(text) > 1 and text[1] == ":"):
        return text
    return str(expanded.resolve())


def service_command(settings_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", f"{IMPORT_PACKAGE}.cli"]
    if settings_path:
        command.extend(["--settings", service_settings_path(settings_path)])
    command.append("serve")
    return command


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
