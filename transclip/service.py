from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from .asr import ASRBackend, build_asr_backend
from .audio import AudioRecorder
from .cleanup import CleanupBackend, build_cleanup_backend
from .debug_capture import DebugCapture
from .dictation_session import DictationSession
from .history import append_transcript_history
from .keyword_restore import restore_keywords
from .service_routes import dispatch_get, dispatch_post
from .settings import Settings, load_settings


class InferenceEngine:
    def __init__(
        self,
        settings: Settings,
        asr_backend: ASRBackend | None = None,
        cleanup_backend: CleanupBackend | None = None,
    ):
        self.settings = settings
        self.asr_backend = asr_backend or build_asr_backend(settings)
        self.cleanup_backend = cleanup_backend or build_cleanup_backend(settings)
        self.debug_capture = DebugCapture(settings)
        self.dictation_session = DictationSession(
            settings,
            transcribe=self._transcribe_for_session,
            recorder_factory=lambda current_settings: AudioRecorder(current_settings),
        )

    def health(self) -> dict[str, Any]:
        status = self.dictation_session.status()
        return {
            "status": status,
            "asr_backend": self.asr_backend.name,
            "asr_model": self.asr_backend.model,
            "cleanup_backend": self.cleanup_backend.name,
            "cleanup_model": self.settings.cleanup_model,
            "cleanup_enabled": self.settings.cleanup_enabled,
            "language": self.settings.language,
            "max_recording_seconds": self.settings.max_recording_seconds,
            "min_recording_ms": self.settings.min_recording_ms,
            "toggle_cooldown_ms": self.settings.toggle_cooldown_ms,
            "hotkey": self.settings.active_hotkey,
            "paste_shortcut": self.settings.paste_shortcut,
            "clipboard_restore_delay_ms": self.settings.clipboard_restore_delay_ms,
            "restore_clipboard_after_paste": self.settings.restore_clipboard_after_paste,
        }

    def start_recording(self) -> dict[str, Any]:
        return self.dictation_session.start_recording()

    def stop_recording(
        self,
        cleanup: bool | None = None,
        discard: bool = False,
        source: str = "/record/stop",
        record_history: bool = False,
    ) -> dict[str, Any]:
        result = self.dictation_session.stop_recording(
            cleanup=cleanup,
            discard=discard,
            source=source,
        )
        if record_history:
            _append_transcript_history(result, self.settings, source=source, duration_ms=result.get("duration_ms"))
        return result

    def toggle_recording(
        self,
        cleanup: bool | None = None,
        record_history: bool = False,
    ) -> dict[str, Any]:
        result = self.dictation_session.toggle_recording(cleanup=cleanup)
        if record_history:
            _append_transcript_history(
                result,
                self.settings,
                source="/record/toggle",
                duration_ms=result.get("duration_ms"),
            )
        return result

    def cleanup_text(self, text: str) -> dict[str, Any]:
        result = self.cleanup_backend.cleanup(text)
        return asdict(result)

    def transcribe(
        self,
        wav_path: Path,
        cleanup: bool | None = None,
        source: str = "/transcribe",
        record_history: bool = False,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        start = perf_counter()
        asr_result = self.asr_backend.transcribe(wav_path, keywords=keywords)
        should_cleanup = self.settings.cleanup_enabled if cleanup is None else cleanup
        raw_asr = restore_keywords(asr_result.text, keywords or [])
        cleaned = raw_asr
        cleanup_result = None
        timings = dict(asr_result.timings_ms)
        if should_cleanup:
            cleanup_result = self.cleanup_backend.cleanup(raw_asr)
            cleaned = cleanup_result.text
            timings.update(cleanup_result.timings_ms)
        timings["end_to_end"] = round((perf_counter() - start) * 1000, 3)
        capture_dir = self.debug_capture.write(
            wav_path=wav_path,
            raw_asr=asr_result.text,
            cleaned=cleaned,
            timings=timings,
            model_versions={
                "asr_backend": asr_result.backend,
                "asr_model": asr_result.model,
                "cleanup_backend": self.cleanup_backend.name,
                "cleanup_model": self.settings.cleanup_model,
            },
        )
        result = {
            "text": cleaned,
            "raw_asr": raw_asr,
            "cleanup": asdict(cleanup_result) if cleanup_result else None,
            "timings_ms": timings,
            "debug_capture_dir": str(capture_dir) if capture_dir else None,
            "asr_backend": asr_result.backend,
            "asr_model": asr_result.model,
            "cleanup_backend": self.cleanup_backend.name,
            "cleanup_enabled": should_cleanup,
        }
        if record_history:
            _append_transcript_history(result, self.settings, source=source)
        return result

    def _transcribe_for_session(
        self,
        wav_path: Path,
        cleanup: bool | None,
        source: str,
    ) -> dict[str, Any]:
        return self.transcribe(
            wav_path,
            cleanup=cleanup,
            source=source,
            record_history=False,
        )


def create_server(
    settings: Settings | None = None,
    engine: InferenceEngine | None = None,
) -> ThreadingHTTPServer:
    settings = settings or load_settings()
    engine = engine or InferenceEngine(settings)

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            response = dispatch_get(engine, urlparse(self.path).path)
            self._json(response.status, response.payload)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._read_json()
                response = dispatch_post(engine, path, body)
                self._json(response.status, response.payload)
            except Exception as exc:
                capture_dir = engine.debug_capture.write_error(
                    "http_request",
                    exc,
                    {"path": path},
                )
                payload = {"error": str(exc)}
                if capture_dir:
                    payload["debug_capture_dir"] = str(capture_dir)
                self._json(500, payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._cors()
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _cors(self) -> None:
            self.send_header("access-control-allow-origin", "*")
            self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
            self.send_header("access-control-allow-headers", "content-type")

    return ThreadingHTTPServer((settings.host, settings.port), Handler)


def run_server(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    server = create_server(settings)
    print(f"transclip service listening on http://{settings.host}:{settings.port}", flush=True)
    server.serve_forever()


def _append_transcript_history(result: dict[str, Any], settings: Settings, **kwargs: Any) -> None:
    try:
        append_transcript_history(result, settings, **kwargs)
    except Exception as exc:
        result["history_error"] = str(exc)
