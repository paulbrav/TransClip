from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.error import URLError

from .client import InferenceClient
from .daemon_lifecycle import (
    SERVICE_NAME,
    CommandResult,
    Runner,
    build_systemd_unit,
    install_daemon,
    install_linux_daemon,
    logs_dir,
    service_action,
    service_state,
    toggle_log_path,
    uninstall_daemon,
)
from .gnome_shortcut import get_gnome_shortcut_status
from .paste import SystemClipboard, SystemPasteInjector, clipboard_capability, paste_capability
from .platform_runtime import PlatformRuntime, get_runtime
from .settings import Settings

__all__ = [
    "SERVICE_NAME",
    "append_toggle_log",
    "build_systemd_unit",
    "collect_status",
    "install_daemon",
    "install_linux_daemon",
    "last_toggle_log_event",
    "logs_dir",
    "run_smoke_test",
    "service_action",
    "service_state",
    "stream_logs",
    "toggle_log_path",
    "uninstall_daemon",
]


def append_toggle_log(event: dict[str, Any], path: Path | None = None) -> None:
    path = path or toggle_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def last_toggle_log_event(path: Path | None = None) -> dict[str, Any] | None:
    path = path or toggle_log_path()
    if not path.exists():
        return None
    last = _last_nonempty_line(path)
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"unparsed": last}


def _last_nonempty_line(path: Path, chunk_size: int = 8192) -> str | None:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        remainder = b""
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size) + remainder
            lines = chunk.splitlines()
            if position > 0:
                remainder = lines[0] if lines else chunk
                lines = lines[1:]
            for line in reversed(lines):
                stripped = line.strip()
                if stripped:
                    return stripped.decode("utf-8", errors="replace")
        stripped = remainder.strip()
        if stripped:
            return stripped.decode("utf-8", errors="replace")
    return None


def collect_status(
    settings: Settings,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> dict[str, Any]:
    platform_runtime = get_runtime(runtime)
    service = service_state(runner=runner, runtime=platform_runtime)
    try:
        health: dict[str, Any] = InferenceClient(settings).health()
    except URLError as exc:
        health = {"ok": False, "error": str(exc)}
    except Exception as exc:
        health = {"ok": False, "error": str(exc)}

    shortcut = None
    if platform_runtime.system() == "Linux":
        try:
            shortcut = asdict(get_gnome_shortcut_status())
        except Exception as exc:
            shortcut = {"error": str(exc)}

    clipboard = _clipboard_status(platform_runtime)
    paste = _paste_status(platform_runtime)
    last_event = last_toggle_log_event()
    ready = service.get("active") is True and health.get("status") in {"ready", "recording"}
    return {
        "ready": ready,
        "service": service,
        "health": health,
        "shortcut": shortcut,
        "clipboard": clipboard,
        "paste": paste,
        "last_log_event": last_event,
    }


def stream_logs(
    follow: bool,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> int:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system == "Linux" and platform_runtime.which("journalctl"):
        command = ["journalctl", "--user", "-u", SERVICE_NAME, "-n", "80", "--no-pager"]
        if follow:
            command.append("-f")
        runner(command, check=False)
    elif system == "Darwin":
        for path in (logs_dir() / "service.out.log", logs_dir() / "service.err.log"):
            if path.exists():
                print(f"==> {path}")
                print(path.read_text(encoding="utf-8", errors="replace")[-8000:])
    path = toggle_log_path()
    if path.exists():
        print(f"==> {path}")
        print(path.read_text(encoding="utf-8", errors="replace")[-8000:])
    else:
        print(f"no toggle log at {path}")
    return 0


def run_smoke_test(
    settings: Settings,
    paste: bool = False,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    platform_runtime = get_runtime(runtime)
    results: list[CommandResult] = []
    client = InferenceClient(settings)
    try:
        health = client.health()
        results.append(CommandResult(True, f"/health status={health.get('status')}"))
    except Exception as exc:
        results.append(CommandResult(False, f"/health failed: {exc}"))
        return results

    try:
        started = client.record_toggle()
        results.append(
            CommandResult(
                started.get("action") == "started",
                f"/record/toggle action={started.get('action')}",
            )
        )
        stopped = client.record_stop(discard=True)
        results.append(CommandResult(stopped.get("discarded") is True, "/record/stop discard=true"))
    except Exception as exc:
        results.append(CommandResult(False, f"record toggle/stop failed: {exc}"))

    try:
        clipboard = SystemClipboard()
        prior = None
        try:
            prior = clipboard.read()
        except Exception:
            prior = None
        marker = "transclip-smoke-test"
        clipboard.write(marker)
        roundtrip = clipboard.read()
        if prior is not None:
            clipboard.write(prior)
        results.append(CommandResult(roundtrip == marker, f"clipboard backend={clipboard.backend_name}"))
    except Exception as exc:
        results.append(CommandResult(False, f"clipboard round-trip failed: {exc}"))

    capability = paste_capability(runtime=platform_runtime)
    results.append(
        CommandResult(
            capability.ok,
            f"paste backend={capability.backend or 'missing'}; {capability.detail}",
        )
    )

    if platform_runtime.system() == "Linux":
        shortcut = get_gnome_shortcut_status()
        results.append(
            CommandResult(
                shortcut.installed and shortcut.command_exists,
                "shortcut "
                f"installed={shortcut.installed} "
                f"binding={shortcut.binding} "
                f"command_exists={shortcut.command_exists}",
            )
        )

    if paste:
        input("Focus a text editor, then press Enter to paste known smoke-test text...")
        try:
            clipboard = SystemClipboard()
            clipboard.write("transclip interactive paste smoke")
            pasted = SystemPasteInjector().paste()
            answer = input("Did the text appear in the focused app? [y/N] ").strip().lower()
            results.append(
                CommandResult(
                    pasted and answer == "y",
                    f"interactive paste command={pasted} user_confirmed={answer == 'y'}",
                )
            )
        except Exception as exc:
            results.append(CommandResult(False, f"interactive paste failed: {exc}"))
    return results


def _clipboard_status(runtime: PlatformRuntime | None = None) -> dict[str, Any]:
    capability = clipboard_capability(runtime=runtime)
    if capability.ok:
        return {"ok": True, "backend": capability.backend}
    return {"ok": False, "error": capability.detail}


def _paste_status(runtime: PlatformRuntime | None = None) -> dict[str, Any]:
    capability = paste_capability(runtime=runtime)
    return {"ok": capability.ok, "backend": capability.backend, "detail": capability.detail}
