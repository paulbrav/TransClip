from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from typing import Any

from transclip.daemon import toggle_log_path
from transclip.daemon.common import CommandResult
from transclip.desktop.paste import SystemClipboard
from transclip.models import ModelRow


def print_command_results(results: Iterable[CommandResult]) -> None:
    for result in results:
        status = "ok" if result.ok else "failed"
        print(f"{status}\t{result.detail}")


def format_status(status: dict[str, Any]) -> str:
    lines: list[str] = [f"state\t{'ready' if status['ready'] else 'not-ready'}"]
    service = status["service"]
    lines.append(
        "service\t" + f"installed={service.get('installed')} active={service.get('active')} {service.get('detail')}"
    )
    health = status["health"]
    lines.append("health\t" + json.dumps(health, sort_keys=True))
    clipboard = status["clipboard"]
    lines.append("clipboard\t" + json.dumps(clipboard, sort_keys=True))
    paste = status["paste"]
    lines.append("paste\t" + json.dumps(paste, sort_keys=True))
    shortcut = status.get("shortcut")
    if shortcut is not None:
        lines.append("shortcut\t" + json.dumps(shortcut, sort_keys=True))
    last_event = status.get("last_log_event")
    lines.append(
        "last_log_event\t" + (json.dumps(last_event, sort_keys=True) if last_event else f"none at {toggle_log_path()}")
    )
    return "\n".join(lines)


def format_history(events: list[dict[str, Any]]) -> str:
    lines = []
    for index, event in enumerate(events, start=1):
        timestamp = event.get("timestamp", "")
        source = event.get("source", "")
        text = str(event.get("text", "")).replace("\n", " ")
        lines.append(f"{index}\t{timestamp}\t{source}\t{text}")
    return "\n".join(lines)


def copy_history(events: list[dict[str, Any]], index: int) -> int:
    if index < 1 or index > len(events):
        print(f"history entry {index} is not in the displayed range", file=sys.stderr)
        return 1
    text = str(events[index - 1].get("text") or "")
    try:
        SystemClipboard().write(text)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"copied\thistory {index}")
    return 0


def format_model_rows(rows: list[ModelRow]) -> str:
    lines = ["model_id\tbackend\tmarker\tcached\tcache_path"]
    for row in rows:
        lines.append(f"{row.model_id}\t{row.backend}\t{row.marker}\t{row.cached}\t{row.cache_path}")
    return "\n".join(lines)
