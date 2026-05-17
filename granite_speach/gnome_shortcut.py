from __future__ import annotations

import ast
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

GNOME_MEDIA_KEYS_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
GNOME_CUSTOM_KEYBINDINGS_KEY = "custom-keybindings"
GNOME_CUSTOM_KEYBINDING_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
GRANITE_SHORTCUT_NAME = "Granite Speach Toggle"
GRANITE_SHORTCUT_BINDING = "<Super><Shift>XF86TouchpadOff"
GRANITE_SHORTCUT_PATH = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/granite-speach-toggle/"

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(slots=True)
class GnomeShortcutStatus:
    installed: bool
    path: str | None
    name: str | None
    binding: str | None
    command: str | None
    command_exists: bool


@dataclass(slots=True)
class GnomeShortcutInstallResult:
    path: str
    name: str
    binding: str
    command: str


def build_toggle_command(settings_path: Path | None = None) -> str:
    command = build_toggle_invocation(settings_path)
    script = (
        'mkdir -p "$HOME/.cache/granite-speach"; '
        + shlex.join(command)
        + ' >> "$HOME/.cache/granite-speach/toggle-record.log" 2>&1'
    )
    return shlex.join(["/bin/sh", "-lc", script])


def build_toggle_invocation(settings_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", "granite_speach.cli"]
    if settings_path:
        command.extend(["--settings", str(settings_path.expanduser().resolve())])
    command.extend(["toggle-record", "--paste"])
    return command


def install_gnome_shortcut(
    command: str,
    binding: str = GRANITE_SHORTCUT_BINDING,
    runner: Runner = subprocess.run,
) -> GnomeShortcutInstallResult:
    _require_gsettings()
    paths = get_custom_keybinding_paths(runner=runner)
    path = _find_granite_path(paths, runner=runner) or GRANITE_SHORTCUT_PATH
    if path not in paths:
        paths.append(path)
        _gsettings_set(
            GNOME_MEDIA_KEYS_SCHEMA,
            GNOME_CUSTOM_KEYBINDINGS_KEY,
            _format_string_array(paths),
            runner=runner,
        )
    _gsettings_set_relocatable(path, "name", GRANITE_SHORTCUT_NAME, runner=runner)
    _gsettings_set_relocatable(path, "command", command, runner=runner)
    _gsettings_set_relocatable(path, "binding", binding, runner=runner)
    return GnomeShortcutInstallResult(
        path=path,
        name=GRANITE_SHORTCUT_NAME,
        binding=binding,
        command=command,
    )


def get_gnome_shortcut_status(
    runner: Runner = subprocess.run,
) -> GnomeShortcutStatus:
    if not shutil.which("gsettings"):
        return GnomeShortcutStatus(False, None, None, None, None, False)
    paths = get_custom_keybinding_paths(runner=runner)
    path = _find_granite_path(paths, runner=runner)
    if not path:
        return GnomeShortcutStatus(False, None, None, None, None, False)
    name = _gsettings_get_relocatable(path, "name", runner=runner)
    binding = _gsettings_get_relocatable(path, "binding", runner=runner)
    command = _gsettings_get_relocatable(path, "command", runner=runner)
    return GnomeShortcutStatus(
        installed=True,
        path=path,
        name=name,
        binding=binding,
        command=command,
        command_exists=command_exists(command),
    )


def get_custom_keybinding_paths(runner: Runner = subprocess.run) -> list[str]:
    raw = _gsettings_get(
        GNOME_MEDIA_KEYS_SCHEMA,
        GNOME_CUSTOM_KEYBINDINGS_KEY,
        runner=runner,
    )
    return _parse_string_array(raw)


def command_exists(command: str | None) -> bool:
    if not command:
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if not parts:
        return False
    program = parts[0]
    if os.path.isabs(program):
        return os.access(program, os.X_OK)
    return shutil.which(program) is not None


def _require_gsettings() -> None:
    if not shutil.which("gsettings"):
        raise RuntimeError("gsettings is required to install the GNOME shortcut")


def _find_granite_path(paths: list[str], runner: Runner) -> str | None:
    if GRANITE_SHORTCUT_PATH in paths:
        return GRANITE_SHORTCUT_PATH
    for path in paths:
        if _gsettings_get_relocatable(path, "name", runner=runner) == GRANITE_SHORTCUT_NAME:
            return path
    return None


def _gsettings_get(schema: str, key: str, runner: Runner) -> str:
    result = runner(
        ["gsettings", "get", schema, key],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _gsettings_set(schema: str, key: str, value: str, runner: Runner) -> None:
    runner(
        ["gsettings", "set", schema, key, value],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def _gsettings_get_relocatable(path: str, key: str, runner: Runner) -> str | None:
    try:
        raw = _gsettings_get(f"{GNOME_CUSTOM_KEYBINDING_SCHEMA}:{path}", key, runner)
    except subprocess.CalledProcessError:
        return None
    return _parse_string(raw)


def _gsettings_set_relocatable(
    path: str,
    key: str,
    value: str,
    runner: Runner,
) -> None:
    _gsettings_set(
        f"{GNOME_CUSTOM_KEYBINDING_SCHEMA}:{path}",
        key,
        value,
        runner=runner,
    )


def _parse_string_array(raw: str) -> list[str]:
    value = raw.strip()
    if value.startswith("@as "):
        value = value[4:].strip()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _parse_string(raw: str) -> str | None:
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return str(parsed) if isinstance(parsed, str) else None


def _format_string_array(values: list[str]) -> str:
    return "[" + ", ".join(_quote(value) for value in values) + "]"


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
