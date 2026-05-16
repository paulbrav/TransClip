from __future__ import annotations

from dataclasses import asdict
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Lock
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from .asr import ASRBackend, build_asr_backend
from .audio import AudioRecorder
from .cleanup import CleanupBackend, build_cleanup_backend
from .debug_capture import DebugCapture
from .glossary import load_keywords
from .settings import Settings, keywords_path, load_settings


class InferenceEngine:
    def __init__(
        self,
        settings: Settings,
        asr_backend: ASRBackend | None = None,
        cleanup_backend: CleanupBackend | None = None,
        keywords: list[str] | None = None,
        keyword_path: Path | None = None,
    ):
        self.settings = settings
        self.asr_backend = asr_backend or build_asr_backend(settings)
        self.cleanup_backend = cleanup_backend or build_cleanup_backend(settings)
        self.keywords = keywords if keywords is not None else load_keywords(keyword_path or keywords_path())
        self.debug_capture = DebugCapture(settings)
        self._recording_lock = Lock()
        self._recorder: AudioRecorder | None = None
        self._recording_started_at = 0.0

    def health(self) -> dict[str, Any]:
        with self._recording_lock:
            status = "recording" if self._recorder else "ready"
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
            "hotkey": self.settings.active_hotkey,
            "paste_shortcut": self.settings.paste_shortcut,
            "clipboard_restore_delay_ms": self.settings.clipboard_restore_delay_ms,
            "restore_clipboard_after_paste": self.settings.restore_clipboard_after_paste,
        }

    def start_recording(self) -> dict[str, Any]:
        with self._recording_lock:
            if self._recorder is not None:
                return {"status": "recording", "already_recording": True}
            recorder = AudioRecorder(self.settings)
            recorder.start()
            self._recorder = recorder
            self._recording_started_at = perf_counter()
        return {"status": "recording", "already_recording": False}

    def stop_recording(
        self,
        cleanup: bool | None = None,
        keywords: list[str] | None = None,
        discard: bool = False,
    ) -> dict[str, Any]:
        with self._recording_lock:
            if self._recorder is None:
                raise RuntimeError("Recorder is not running")
            recorder = self._recorder
            started_at = self._recording_started_at
            self._recorder = None
            self._recording_started_at = 0.0
        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = recorder.stop_to_wav(Path(tmp) / "recording.wav")
            if discard:
                return {"status": "ready", "duration_ms": duration_ms, "discarded": True}
            result = self.transcribe(wav_path, cleanup=cleanup, keywords=keywords)
        result["duration_ms"] = duration_ms
        return result

    def cleanup_text(self, text: str, keywords: list[str] | None = None) -> dict[str, Any]:
        active_keywords = self.keywords if keywords is None else keywords
        result = self.cleanup_backend.cleanup(text, active_keywords)
        return asdict(result)

    def transcribe(
        self,
        wav_path: Path,
        cleanup: bool | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        start = perf_counter()
        active_keywords = self.keywords if keywords is None else keywords
        asr_result = self.asr_backend.transcribe(wav_path, active_keywords)
        should_cleanup = self.settings.cleanup_enabled if cleanup is None else cleanup
        cleaned = asr_result.text
        cleanup_result = None
        timings = dict(asr_result.timings_ms)
        if should_cleanup:
            cleanup_result = self.cleanup_backend.cleanup(asr_result.text, active_keywords)
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
        return {
            "text": cleaned,
            "raw_asr": asr_result.text,
            "cleanup": asdict(cleanup_result) if cleanup_result else None,
            "timings_ms": timings,
            "debug_capture_dir": str(capture_dir) if capture_dir else None,
        }


def create_server(
    settings: Settings | None = None,
    engine: InferenceEngine | None = None,
    keyword_path: Path | None = None,
) -> ThreadingHTTPServer:
    settings = settings or load_settings()
    engine = engine or InferenceEngine(settings, keyword_path=keyword_path)
    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            if urlparse(self.path).path == "/health":
                self._json(200, engine.health())
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._read_json()
                if path == "/cleanup":
                    self._json(
                        200,
                        engine.cleanup_text(
                            str(body.get("text", "")),
                            keywords=_keywords_from_request(body),
                        ),
                    )
                    return
                if path == "/record/start":
                    self._json(200, engine.start_recording())
                    return
                if path == "/record/stop":
                    self._json(
                        200,
                        engine.stop_recording(
                            cleanup=body.get("cleanup"),
                            keywords=_keywords_from_request(body),
                            discard=bool(body.get("discard")),
                        ),
                    )
                    return
                if path in {"/transcribe", "/cleanup/transcribe"}:
                    wav_path = _wav_from_request(body)
                    cleanup = True if path == "/cleanup/transcribe" else body.get("cleanup")
                    self._json(
                        200,
                        engine.transcribe(
                            wav_path,
                            cleanup=cleanup,
                            keywords=_keywords_from_request(body),
                        ),
                    )
                    return
                self._json(404, {"error": "not found"})
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


def run_server(settings: Settings | None = None, keyword_path: Path | None = None) -> None:
    settings = settings or load_settings()
    server = create_server(settings, keyword_path=keyword_path)
    print(f"granite-speach service listening on http://{settings.host}:{settings.port}", flush=True)
    server.serve_forever()


def _wav_from_request(body: dict[str, Any]) -> Path:
    if "audio_path" in body:
        path = Path(body["audio_path"]).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if "audio_base64" in body:
        data = base64.b64decode(body["audio_base64"])
        ext = str(body.get("audio_ext", "wav")).strip().lstrip(".") or "wav"
        handle = tempfile.NamedTemporaryFile(prefix="granite-speach-", suffix=f".{ext}", delete=False)
        with handle:
            handle.write(data)
        return Path(handle.name)
    raise ValueError("Request must include audio_path or audio_base64")


def _keywords_from_request(body: dict[str, Any]) -> list[str] | None:
    keywords = body.get("keywords")
    if keywords is None:
        return None
    if not isinstance(keywords, list):
        raise ValueError("keywords must be a list of strings")
    return [str(keyword) for keyword in keywords]
