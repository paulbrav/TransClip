from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RouteResponse:
    status: int
    payload: dict[str, Any]


def dispatch_get(engine, path: str) -> RouteResponse:
    if path == "/health":
        return RouteResponse(200, engine.health())
    return RouteResponse(404, {"error": "not found"})


def dispatch_post(engine, path: str, body: dict[str, Any]) -> RouteResponse:
    if path == "/cleanup":
        return RouteResponse(
            200,
            engine.cleanup_text(str(body.get("text", ""))),
        )
    if path == "/record/start":
        return RouteResponse(200, engine.start_recording())
    if path == "/record/stop":
        return RouteResponse(
            200,
            engine.stop_recording(
                cleanup=body.get("cleanup"),
                discard=bool(body.get("discard")),
                source="/record/stop",
                record_history=True,
            ),
        )
    if path == "/record/toggle":
        return RouteResponse(
            200,
            engine.toggle_recording(
                cleanup=body.get("cleanup"),
                record_history=True,
            ),
        )
    if path in {"/transcribe", "/cleanup/transcribe"}:
        cleanup = True if path == "/cleanup/transcribe" else body.get("cleanup")
        wav_path = wav_from_request(body)
        try:
            return RouteResponse(
                200,
                engine.transcribe(
                    wav_path,
                    cleanup=cleanup,
                    source=path,
                    record_history=True,
                ),
            )
        finally:
            if "audio_base64" in body:
                wav_path.unlink(missing_ok=True)
    return RouteResponse(404, {"error": "not found"})


def wav_from_request(body: dict[str, Any]) -> Path:
    if "audio_path" in body:
        path = Path(body["audio_path"]).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if "audio_base64" in body:
        data = base64.b64decode(body["audio_base64"])
        ext = str(body.get("audio_ext", "wav")).strip().lstrip(".") or "wav"
        with tempfile.NamedTemporaryFile(prefix="transclip-", suffix=f".{ext}", delete=False) as handle:
            handle.write(data)
            return Path(handle.name)
    raise ValueError("Request must include audio_path or audio_base64")
