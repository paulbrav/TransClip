from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Protocol

from transclip.audio import AudioRecorder
from transclip.settings import Settings


class Recorder(Protocol):
    def start(self) -> None: ...

    def stop_to_wav(self, output_path: Path) -> Path: ...


RecorderFactory = Callable[[Settings], Recorder]
Transcriber = Callable[[Path, bool | None, str], dict[str, Any]]
Clock = Callable[[], float]


class DictationSession:
    def __init__(
        self,
        settings: Settings,
        transcribe: Transcriber,
        recorder_factory: RecorderFactory | None = None,
        clock: Clock = perf_counter,
    ):
        self.settings = settings
        self._transcribe = transcribe
        self._recorder_factory = recorder_factory or AudioRecorder
        self._clock = clock
        self._lock = Lock()
        self._recorder: Recorder | None = None
        self._recording_started_at = 0.0
        self._last_toggle_accepted_at = 0.0

    def status(self) -> str:
        with self._lock:
            return "recording" if self._recorder else "ready"

    def start_recording(self) -> dict[str, Any]:
        with self._lock:
            if self._recorder is not None:
                return {"status": "recording", "already_recording": True}
            recorder = self._recorder_factory(self.settings)
            recorder.start()
            self._recorder = recorder
            self._recording_started_at = self._clock()
        return {"status": "recording", "already_recording": False}

    def stop_recording(
        self,
        cleanup: bool | None = None,
        discard: bool = False,
        source: str = "/record/stop",
    ) -> dict[str, Any]:
        with self._lock:
            if self._recorder is None:
                raise RuntimeError("Recorder is not running")
            recorder = self._recorder
            started_at = self._recording_started_at
            self._recorder = None
            self._recording_started_at = 0.0
        return self._finish_recording(
            recorder,
            started_at,
            cleanup=cleanup,
            discard=discard,
            source=source,
        )

    def toggle_recording(
        self,
        cleanup: bool | None = None,
    ) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            cooldown_seconds = max(0, self.settings.toggle_cooldown_ms) / 1000
            if (
                cooldown_seconds
                and self._last_toggle_accepted_at
                and now - self._last_toggle_accepted_at < cooldown_seconds
            ):
                return {
                    "status": "recording" if self._recorder is not None else "ready",
                    "action": "ignored",
                    "reason": "toggle_cooldown",
                    "cooldown_ms": self.settings.toggle_cooldown_ms,
                }
            self._last_toggle_accepted_at = now
            if self._recorder is None:
                recorder = self._recorder_factory(self.settings)
                recorder.start()
                self._recorder = recorder
                self._recording_started_at = now
                return {"status": "recording", "action": "started", "already_recording": False}

            recorder = self._recorder
            started_at = self._recording_started_at
            self._recorder = None
            self._recording_started_at = 0.0

        duration_ms = (self._clock() - started_at) * 1000
        discard = duration_ms < self.settings.min_recording_ms
        result = self._finish_recording(
            recorder,
            started_at,
            cleanup=cleanup,
            discard=discard,
            source="/record/toggle",
        )
        result["status"] = "ready"
        result["action"] = "discarded" if discard else "stopped"
        return result

    def _finish_recording(
        self,
        recorder: Recorder,
        started_at: float,
        *,
        cleanup: bool | None,
        discard: bool,
        source: str,
    ) -> dict[str, Any]:
        duration_ms = round((self._clock() - started_at) * 1000, 3)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = recorder.stop_to_wav(Path(tmp) / "recording.wav")
            if discard:
                return {"status": "ready", "duration_ms": duration_ms, "discarded": True}
            result = self._transcribe(wav_path, cleanup, source)
        result["duration_ms"] = duration_ms
        return result
