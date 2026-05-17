from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .settings import Settings


def history_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Logs" / "granite-speach"
    return Path.home() / ".cache" / "granite-speach"


def history_path() -> Path:
    return history_dir() / "history.jsonl"


def append_history_event(event: dict[str, Any], path: Path | None = None) -> None:
    text = str(event.get("text") or "").strip()
    if not text:
        return
    payload = dict(event)
    payload["text"] = text
    payload.setdefault("timestamp", timestamp())
    path = path or history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_transcript_history(
    result: dict[str, Any],
    settings: Settings,
    source: str,
    duration_ms: float | None = None,
    path: Path | None = None,
) -> None:
    text = str(result.get("text") or "").strip()
    if not text:
        return
    event: dict[str, Any] = {
        "timestamp": timestamp(),
        "text": text,
        "raw_asr": str(result.get("raw_asr") or ""),
        "source": source,
        "asr_backend": result.get("asr_backend") or settings.asr_backend,
        "asr_model": result.get("asr_model") or settings.asr_model,
        "cleanup_backend": result.get("cleanup_backend") or settings.cleanup_runtime,
        "cleanup_enabled": bool(result.get("cleanup_enabled", settings.cleanup_enabled)),
    }
    if duration_ms is None:
        duration_ms = result.get("duration_ms")
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    append_history_event(event, path=path)


def read_history(limit: int | None = None, path: Path | None = None) -> list[dict[str, Any]]:
    path = path or history_path()
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    events.reverse()
    if limit is not None:
        return events[:limit]
    return events


def timestamp() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")
