from __future__ import annotations

import plistlib
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path, PurePath

from transclip.daemon.common import CommandResult, logs_dir, repo_root
from transclip.paths import service_settings_path
from transclip.platform.runtime import PlatformRuntime, get_runtime, user_cache_dir
from transclip.product import CACHE_DIR_NAME, IMPORT_PACKAGE
from transclip.settings import Settings

Runner = Callable[..., subprocess.CompletedProcess[str]]

HOTKEY_APP_NAME = "TransClipHotkey"
HOTKEY_BUNDLE_ID = "com.paulbrav.TransClipHotkey"
HOTKEY_LAUNCHD_LABEL = "com.paulbrav.transclip-hotkey"
HOTKEY_LOG_NAME = "hotkey.log"
TOGGLE_WRAPPER_NAME = "transclip-toggle"
DEFAULT_STOP_TIMEOUT_SECONDS = 75
STALE_LOCK_SECONDS = 90


@dataclass(frozen=True, slots=True)
class MacOSHotkeyInstall:
    app_path: Path
    launch_agent_path: Path
    wrapper_path: Path
    source_path: Path


def macos_hotkey_app_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Applications" / f"{HOTKEY_APP_NAME}.app"


def macos_hotkey_launch_agent_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Library" / "LaunchAgents" / f"{HOTKEY_LAUNCHD_LABEL}.plist"


def macos_toggle_wrapper_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "bin" / TOGGLE_WRAPPER_NAME


def macos_hotkey_source_path(runtime: PlatformRuntime | None = None) -> Path:
    return user_cache_dir(CACHE_DIR_NAME, runtime) / "macos-hotkey" / f"{HOTKEY_APP_NAME}.swift"


def macos_hotkey_log_path(runtime: PlatformRuntime | None = None) -> Path:
    return logs_dir(runtime) / HOTKEY_LOG_NAME


def macos_hotkey_state_path(runtime: PlatformRuntime | None = None) -> Path:
    return logs_dir(runtime) / "hotkey-state.tsv"


def macos_hotkey_target(runtime: PlatformRuntime | None = None) -> str:
    return f"{macos_launchd_gui_domain(runtime)}/{HOTKEY_LAUNCHD_LABEL}"


def macos_launchd_gui_domain(runtime: PlatformRuntime | None = None) -> str:
    output = get_runtime(runtime).check_output(["id", "-u"])
    if isinstance(output, bytes):
        output = output.decode()
    return f"gui/{output.strip()}"


def build_macos_toggle_wrapper(
    settings: Settings,
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
    *,
    stop_timeout_seconds: int = DEFAULT_STOP_TIMEOUT_SECONDS,
    stale_lock_seconds: int = STALE_LOCK_SECONDS,
) -> str:
    log_path = logs_dir(runtime) / "toggle-record.log"
    state_path = macos_hotkey_state_path(runtime)
    base_url = f"http://{settings.host}:{settings.port}"
    python = shlex.quote(sys.executable)
    cli = f"{python} -m {shlex.quote(IMPORT_PACKAGE + '.cli')}"
    if settings_path:
        cli += f" --settings {shlex.quote(service_settings_path(settings_path))}"
    restart_command = f'cd {shlex.quote(_macos_path_text(repo_root()))} && {cli} restart >> "$LOG" 2>&1'
    return _render_resource(
        "toggle_wrapper.sh",
        {
            "@@LOG@@": shlex.quote(_macos_path_text(log_path)),
            "@@STATE@@": shlex.quote(_macos_path_text(state_path)),
            "@@BASE@@": shlex.quote(base_url),
            "@@MAX_SECONDS@@": str(int(stop_timeout_seconds)),
            "@@STALE_LOCK_SECONDS@@": str(int(stale_lock_seconds)),
            "@@RESTART_COMMAND@@": restart_command,
            "@@PYTHON@@": python,
        },
    )


def build_macos_hotkey_source(
    wrapper_path: Path,
    log_path: Path,
    state_path: Path,
) -> str:
    return _render_resource(
        "macos_hotkey.swift",
        {
            "@@LOG_PATH@@": _swift_string(_macos_path_text(log_path)),
            "@@WRAPPER_PATH@@": _swift_string(_macos_path_text(wrapper_path)),
            "@@STATE_PATH@@": _swift_string(_macos_path_text(state_path)),
        },
    )


def build_macos_hotkey_launch_agent(runtime: PlatformRuntime | None = None) -> bytes:
    log_root = logs_dir(runtime)
    payload = {
        "Label": HOTKEY_LAUNCHD_LABEL,
        "ProgramArguments": [str(_macos_hotkey_executable_path(runtime))],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_root / "hotkey.out.log"),
        "StandardErrorPath": str(log_root / "hotkey.err.log"),
    }
    return plistlib.dumps(payload, sort_keys=True)


def install_macos_hotkey(
    settings: Settings,
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> tuple[MacOSHotkeyInstall, list[CommandResult]]:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() != "Darwin":
        raise RuntimeError("macOS hotkey helper is only supported on Darwin")

    paths = MacOSHotkeyInstall(
        app_path=macos_hotkey_app_path(platform_runtime),
        launch_agent_path=macos_hotkey_launch_agent_path(platform_runtime),
        wrapper_path=macos_toggle_wrapper_path(platform_runtime),
        source_path=macos_hotkey_source_path(platform_runtime),
    )
    results: list[CommandResult] = []
    logs_dir(platform_runtime).mkdir(parents=True, exist_ok=True)

    paths.wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    paths.wrapper_path.write_text(
        build_macos_toggle_wrapper(settings, settings_path, platform_runtime),
        encoding="utf-8",
    )
    paths.wrapper_path.chmod(0o755)
    results.append(CommandResult(True, f"wrote {paths.wrapper_path}"))

    swiftc = platform_runtime.which("swiftc")
    if not swiftc:
        results.append(
            CommandResult(False, "swiftc missing; install Xcode Command Line Tools with: xcode-select --install")
        )
        return paths, results

    executable = _macos_hotkey_executable_path(platform_runtime)
    executable.parent.mkdir(parents=True, exist_ok=True)
    paths.source_path.parent.mkdir(parents=True, exist_ok=True)
    paths.source_path.write_text(
        build_macos_hotkey_source(
            paths.wrapper_path,
            macos_hotkey_log_path(platform_runtime),
            macos_hotkey_state_path(platform_runtime),
        ),
        encoding="utf-8",
    )
    _macos_hotkey_info_plist_path(platform_runtime).write_bytes(_build_info_plist())
    results.append(CommandResult(True, f"wrote {paths.source_path}"))
    results.append(_run_command([swiftc, str(paths.source_path), "-o", str(executable)], runner))
    if not results[-1].ok:
        return paths, results
    executable.chmod(0o755)

    codesign = platform_runtime.which("codesign") or "codesign"
    results.append(_run_command([codesign, "--force", "--deep", "--sign", "-", str(paths.app_path)], runner))
    if not results[-1].ok:
        return paths, results

    paths.launch_agent_path.parent.mkdir(parents=True, exist_ok=True)
    paths.launch_agent_path.write_bytes(build_macos_hotkey_launch_agent(platform_runtime))
    results.append(CommandResult(True, f"wrote {paths.launch_agent_path}"))
    target = macos_hotkey_target(platform_runtime)
    domain = macos_launchd_gui_domain(platform_runtime)
    results.append(_run_command(["launchctl", "bootout", target], runner, tolerate_failure=True))
    results.append(_run_command(["launchctl", "bootstrap", domain, str(paths.launch_agent_path)], runner))
    results.append(
        CommandResult(
            True,
            ("Grant Accessibility to TransClipHotkey.app so it can observe Option+Space and post Command+V."),
        )
    )
    return paths, results


def uninstall_macos_hotkey(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    platform_runtime = get_runtime(runtime)
    results = [
        _run_command(["launchctl", "bootout", macos_hotkey_target(platform_runtime)], runner, tolerate_failure=True)
    ]
    path = macos_hotkey_launch_agent_path(platform_runtime)
    if path.exists():
        path.unlink()
        results.append(CommandResult(True, f"removed {path}"))
    app_path = macos_hotkey_app_path(platform_runtime)
    if app_path.exists():
        shutil.rmtree(app_path)
        results.append(CommandResult(True, f"removed {app_path}"))
    wrapper_path = macos_toggle_wrapper_path(platform_runtime)
    if wrapper_path.exists():
        wrapper_path.unlink()
        results.append(CommandResult(True, f"removed {wrapper_path}"))
    source_path = macos_hotkey_source_path(platform_runtime)
    if source_path.exists():
        source_path.unlink()
        results.append(CommandResult(True, f"removed {source_path}"))
    return results


def _run_command(command: list[str], runner: Runner, tolerate_failure: bool = False) -> CommandResult:
    try:
        result = runner(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
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


def _build_info_plist() -> bytes:
    return plistlib.dumps(
        {
            "CFBundleExecutable": HOTKEY_APP_NAME,
            "CFBundleIdentifier": HOTKEY_BUNDLE_ID,
            "CFBundleName": HOTKEY_APP_NAME,
            "CFBundleDisplayName": HOTKEY_APP_NAME,
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.0",
            "CFBundleVersion": "1",
            "LSUIElement": True,
        },
        sort_keys=True,
    )


def _macos_hotkey_executable_path(runtime: PlatformRuntime | None = None) -> Path:
    return macos_hotkey_app_path(runtime) / "Contents" / "MacOS" / HOTKEY_APP_NAME


def _macos_hotkey_info_plist_path(runtime: PlatformRuntime | None = None) -> Path:
    return macos_hotkey_app_path(runtime) / "Contents" / "Info.plist"


@cache
def _resource_text(name: str) -> str:
    return (files(__package__) / "resources" / name).read_text(encoding="utf-8")


def _render_resource(name: str, replacements: dict[str, str]) -> str:
    text = _resource_text(name)
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


def _swift_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _macos_path_text(path: PurePath) -> str:
    return path.as_posix()
