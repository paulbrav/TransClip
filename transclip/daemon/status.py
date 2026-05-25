from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transclip.desktop.hotkey import get_gnome_shortcut_status
from transclip.desktop.paste import (
    SystemClipboard,
    SystemPasteInjector,
    available_paste_backend,
    clipboard_capability,
    paste_capability,
    plan_paste_delivery,
)
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.product import SERVICE_NAME
from transclip.service import InferenceClient
from transclip.service.client_health import (
    fetch_service_health_result,
    service_health_check_detail,
    service_health_is_ready,
)
from transclip.settings import Settings

from .common import CommandResult, Runner, logs_dir, toggle_log_path
from .lifecycle import service_state

__all__ = [
    "append_toggle_log",
    "collect_status",
    "last_toggle_log_event",
    "run_smoke_test",
    "stream_logs",
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
    service_state_value = service_state(runner=runner, runtime=platform_runtime)
    service = asdict(service_state_value)
    health_payload, health_error = fetch_service_health_result(settings)
    if health_error is not None:
        health: dict[str, Any] = {"ok": False, "error": health_error}
    else:
        health = health_payload or {"ok": False, "error": "no response"}

    shortcut = None
    if platform_runtime.system() == "Linux":
        try:
            shortcut = asdict(get_gnome_shortcut_status())
        except Exception as exc:
            shortcut = {"error": str(exc)}

    clipboard = _clipboard_status(platform_runtime)
    paste = _paste_status(platform_runtime)
    last_event = last_toggle_log_event()
    ready = service_state_value.active and service_health_is_ready(health)
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
    health, error = fetch_service_health_result(settings)
    if error is not None:
        results.append(CommandResult(False, service_health_check_detail(None, error=error)))
        return results
    results.append(CommandResult(True, service_health_check_detail(health)))

    client = InferenceClient(settings)
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
    elif platform_runtime.system() == "Windows":
        from transclip.settings import active_hotkey

        binding = active_hotkey(settings, platform_runtime)
        results.append(
            CommandResult(
                True,
                f"hotkey configured={binding}; global hotkey requires transclip tray",
            )
        )

    if paste:
        plan = plan_paste_delivery(settings, runtime=platform_runtime)
        backend = available_paste_backend(runtime=platform_runtime)
        print(
            f"Paste smoke: shortcut={plan.label} "
            f"backend={backend or 'unavailable'} "
            f"focused_app_kind={plan.focused_app_kind or 'unknown'}"
        )
        input("Focus a text editor, then press Enter to paste known smoke-test text...")
        try:
            clipboard = SystemClipboard(runtime=platform_runtime)
            clipboard.write("transclip interactive paste smoke")
            injector = SystemPasteInjector(runtime=platform_runtime, shortcut=plan.shortcut)
            pasted = injector.paste()
            answer = input("Did the text appear in the focused app? [y/N] ").strip().lower()
            results.append(
                CommandResult(
                    pasted and answer == "y",
                    f"interactive paste shortcut={plan.label} "
                    f"backend={injector.backend_name or backend or 'unknown'} "
                    f"command={pasted} user_confirmed={answer == 'y'}",
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
