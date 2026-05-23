from __future__ import annotations

import ast
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .platform_capabilities import session_info
from .platform_runtime import PlatformRuntime, get_runtime, user_log_dir
from .product import (
    CLI_COMMAND,
    FALLBACK_HOTKEY_LINUX,
    IMPORT_PACKAGE,
    LEGACY_SHORTCUT_NAME,
    LEGACY_SHORTCUT_PATH,
    LOG_DIR_NAME,
    SHORTCUT_ALT_NAME,
    SHORTCUT_ALT_PATH,
    SHORTCUT_NAME,
    SHORTCUT_PATH,
)
from .settings import DEFAULT_HOTKEY_LINUX, Settings

GNOME_MEDIA_KEYS_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
GNOME_CUSTOM_KEYBINDINGS_KEY = "custom-keybindings"
GNOME_CUSTOM_KEYBINDING_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
TRANSCLIP_SHORTCUT_NAME = SHORTCUT_NAME
TRANSCLIP_SHORTCUT_BINDING = DEFAULT_HOTKEY_LINUX
TRANSCLIP_SHORTCUT_PATH = SHORTCUT_PATH

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


@dataclass(slots=True)
class ShortcutReadiness:
    ok: bool
    detail: str
    status: GnomeShortcutStatus | None = None


def build_toggle_command(
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> str:
    command = build_toggle_invocation(settings_path)
    log_path = toggle_log_shell_path(runtime)
    quoted_log_path = shlex.quote(log_path)
    script = f'mkdir -p "$(dirname {quoted_log_path})"; ' + shlex.join(command) + f" >> {quoted_log_path} 2>&1"
    return shlex.join(["/bin/sh", "-lc", script])


def toggle_log_shell_path(runtime: PlatformRuntime | None = None) -> str:
    log_dir = user_log_dir(LOG_DIR_NAME, runtime)
    return str(log_dir / "toggle-record.log")


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


def build_toggle_invocation(settings_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", f"{IMPORT_PACKAGE}.cli"]
    if settings_path:
        command.extend(["--settings", str(settings_path.expanduser().resolve())])
    command.extend(["toggle-record", "--paste"])
    return command


def install_gnome_shortcut(
    command: str,
    binding: str = TRANSCLIP_SHORTCUT_BINDING,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> GnomeShortcutInstallResult:
    _require_gsettings(runtime)
    paths = _remove_legacy_shortcut_paths(get_custom_keybinding_paths(runner=runner), runner=runner)
    path = _find_transclip_path(paths, runner=runner) or TRANSCLIP_SHORTCUT_PATH
    paths = _ensure_shortcut_path(paths, path)
    paths = _ensure_shortcut_path(paths, SHORTCUT_ALT_PATH)
    _gsettings_set(
        GNOME_MEDIA_KEYS_SCHEMA,
        GNOME_CUSTOM_KEYBINDINGS_KEY,
        _format_string_array(paths),
        runner=runner,
    )
    _gsettings_set_relocatable(path, "name", TRANSCLIP_SHORTCUT_NAME, runner=runner)
    _gsettings_set_relocatable(path, "command", command, runner=runner)
    _gsettings_set_relocatable(path, "binding", binding, runner=runner)
    _gsettings_set_relocatable(SHORTCUT_ALT_PATH, "name", SHORTCUT_ALT_NAME, runner=runner)
    _gsettings_set_relocatable(SHORTCUT_ALT_PATH, "command", command, runner=runner)
    _gsettings_set_relocatable(SHORTCUT_ALT_PATH, "binding", FALLBACK_HOTKEY_LINUX, runner=runner)
    return GnomeShortcutInstallResult(
        path=path,
        name=TRANSCLIP_SHORTCUT_NAME,
        binding=binding,
        command=command,
    )


def get_gnome_shortcut_status(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> GnomeShortcutStatus:
    platform_runtime = get_runtime(runtime)
    if not platform_runtime.which("gsettings"):
        return GnomeShortcutStatus(False, None, None, None, None, False)
    paths = get_custom_keybinding_paths(runner=runner)
    path = _find_transclip_path(paths, runner=runner)
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
        command_exists=command_exists(command, platform_runtime),
    )


def install_shortcut(
    settings_path: Path | None = None,
    binding: str = TRANSCLIP_SHORTCUT_BINDING,
    command: str | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> GnomeShortcutInstallResult:
    return install_gnome_shortcut(
        command=command or build_toggle_command(settings_path),
        binding=binding,
        runner=runner,
        runtime=runtime,
    )


def shortcut_readiness(
    expected_binding: str = TRANSCLIP_SHORTCUT_BINDING,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> ShortcutReadiness:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system == "Darwin":
        return ShortcutReadiness(True, "macOS hotkey helper is not installed by the Python daemon")
    if system != "Linux":
        return ShortcutReadiness(True, f"not checked on {system}")

    info = session_info(runtime=platform_runtime)
    if not platform_runtime.which("gsettings"):
        return ShortcutReadiness(
            False,
            f"session={info.session}; desktop={info.desktop}; GNOME shortcut setup requires gsettings",
        )

    try:
        status = get_gnome_shortcut_status(runner=runner, runtime=platform_runtime)
    except subprocess.CalledProcessError as exc:
        return ShortcutReadiness(
            False,
            f"session={info.session}; desktop={info.desktop}; could not inspect GNOME custom shortcuts: {exc}",
        )

    detail = (
        f"session={info.session}; desktop={info.desktop}; installed={status.installed}; "
        f"binding={status.binding or 'missing'}; command_exists={status.command_exists}"
    )
    if not status.installed:
        return ShortcutReadiness(False, detail + f"; run: {CLI_COMMAND} install-gnome-shortcut", status)
    if status.binding != expected_binding:
        return ShortcutReadiness(False, detail + f"; expected binding={expected_binding}", status)
    if not status.command_exists:
        return ShortcutReadiness(False, detail + f"; command={status.command or 'missing'}", status)
    return ShortcutReadiness(True, detail + f"; command={status.command}", status)


def get_custom_keybinding_paths(runner: Runner = subprocess.run) -> list[str]:
    raw = _gsettings_get(
        GNOME_MEDIA_KEYS_SCHEMA,
        GNOME_CUSTOM_KEYBINDINGS_KEY,
        runner=runner,
    )
    return _parse_string_array(raw)


def command_exists(command: str | None, runtime: PlatformRuntime | None = None) -> bool:
    if not command:
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if not parts:
        return False
    shell_program = Path(parts[0]).name
    if shell_program in {"sh", "bash"} and "-lc" in parts:
        script_index = parts.index("-lc") + 1
        if script_index >= len(parts):
            return False
        return _shell_script_command_exists(parts[script_index], runtime)
    program = parts[0]
    if os.path.isabs(program):
        return os.access(program, os.X_OK)
    return get_runtime(runtime).which(program) is not None


def _shell_script_command_exists(script: str, runtime: PlatformRuntime | None = None) -> bool:
    for segment in reversed(script.split(";")):
        try:
            parts = shlex.split(segment)
        except ValueError:
            continue
        if not parts:
            continue
        program = parts[0]
        if os.path.isabs(program):
            return os.access(program, os.X_OK)
        return get_runtime(runtime).which(program) is not None
    return False


def _require_gsettings(runtime: PlatformRuntime | None = None) -> None:
    if not get_runtime(runtime).which("gsettings"):
        raise RuntimeError("gsettings is required to install the GNOME shortcut")


def _find_transclip_path(paths: list[str], runner: Runner) -> str | None:
    if TRANSCLIP_SHORTCUT_PATH in paths:
        return TRANSCLIP_SHORTCUT_PATH
    for path in paths:
        if _gsettings_get_relocatable(path, "name", runner=runner) == TRANSCLIP_SHORTCUT_NAME:
            return path
    return None


def _ensure_shortcut_path(paths: list[str], path: str) -> list[str]:
    if path in paths:
        return paths
    return [*paths, path]


def _remove_legacy_shortcut_paths(paths: list[str], runner: Runner) -> list[str]:
    kept: list[str] = []
    for path in paths:
        if path in {LEGACY_SHORTCUT_PATH, SHORTCUT_ALT_PATH}:
            continue
        name = _gsettings_get_relocatable(path, "name", runner=runner)
        if name == LEGACY_SHORTCUT_NAME:
            continue
        kept.append(path)
    return kept


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
