from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol

from transclip.asr_streaming import PartialTranscript
from transclip.audio import AudioRecorder
from transclip.settings import Settings

from .streaming import StreamingCapture, StreamingDictationAdapter
from .types import RecordSessionResponse, TranscribeResponse


class Recorder(Protocol):
    def start(self) -> None: ...

    def stop_to_wav(self, output_path: Path) -> Path: ...

    def stop_capture(self) -> None: ...

    def discard(self) -> None: ...


RecorderFactory = Callable[[Settings], Recorder]
Transcriber = Callable[[Path, bool | None, str], TranscribeResponse]
Clock = Callable[[], float]
PhaseCallback = Callable[[], None]


class RecordingHandle(Protocol):
    def start(self) -> None: ...

    def stop(
        self,
        *,
        cleanup: bool | None,
        source: str,
        before_transcribe: PhaseCallback | None = None,
    ) -> TranscribeResponse: ...

    def discard(self) -> None: ...


@dataclass(slots=True)
class BatchRecording:
    recorder: Recorder
    transcribe: Transcriber

    def start(self) -> None:
        self.recorder.start()

    def stop(
        self,
        *,
        cleanup: bool | None,
        source: str,
        before_transcribe: PhaseCallback | None = None,
    ) -> TranscribeResponse:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = self.recorder.stop_to_wav(Path(tmp) / "recording.wav")
            if before_transcribe is not None:
                before_transcribe()
            return self.transcribe(wav_path, cleanup, source)

    def discard(self) -> None:
        self.recorder.discard()


@dataclass(slots=True)
class StreamingRecording:
    adapter: StreamingDictationAdapter
    capture: StreamingCapture
    debug_capture: bool

    def start(self) -> None:
        try:
            self.capture.recorder.start()
        except Exception:
            self.adapter.discard_session(self.capture.session)
            raise

    def stop(
        self,
        *,
        cleanup: bool | None,
        source: str,
        before_transcribe: PhaseCallback | None = None,
    ) -> TranscribeResponse:
        self.adapter.detach_session(self.capture.session)
        try:
            if self.debug_capture:
                with tempfile.TemporaryDirectory() as tmp:
                    wav_path = self.capture.recorder.stop_to_wav(Path(tmp) / "recording.wav")
                    if before_transcribe is not None:
                        before_transcribe()
                    return self.adapter.finish_session(
                        self.capture.session,
                        cleanup,
                        source,
                        wav_path=wav_path,
                    )
            self.capture.recorder.stop_capture()
            if before_transcribe is not None:
                before_transcribe()
            return self.adapter.finish_session(self.capture.session, cleanup, source)
        except Exception:
            self.adapter.discard_session(self.capture.session)
            raise

    def discard(self) -> None:
        try:
            self.capture.recorder.discard()
        finally:
            self.adapter.discard_session(self.capture.session)


@dataclass(slots=True)
class ActiveRecording:
    handle: RecordingHandle
    started_at: float


class DictationSession:
    def __init__(
        self,
        settings: Settings,
        transcribe: Transcriber,
        recorder_factory: RecorderFactory | None = None,
        clock: Clock = perf_counter,
        streaming: StreamingDictationAdapter | None = None,
    ):
        self.settings = settings
        self._transcribe = transcribe
        self._recorder_factory = recorder_factory or AudioRecorder
        self._streaming = streaming
        self._clock = clock
        self._lock = Lock()
        self._active: ActiveRecording | None = None
        self._status = "ready"
        self._last_toggle_accepted_at = 0.0

    def status(self) -> str:
        with self._lock:
            return self._status

    def partial_text(self) -> PartialTranscript:
        if self._streaming is None:
            return PartialTranscript("")
        return self._streaming.partial_text()

    def start_recording(self) -> RecordSessionResponse:
        with self._lock:
            if self._active is not None:
                return {"status": "recording", "already_recording": True}
            handle = self._create_recording()
            handle.start()
            self._active = ActiveRecording(handle, self._clock())
            self._status = "recording"
        return {"status": "recording", "already_recording": False}

    def stop_recording(
        self,
        cleanup: bool | None = None,
        discard: bool = False,
        source: str = "/record/stop",
    ) -> RecordSessionResponse:
        with self._lock:
            active = self._pop_active()
        return self._finish_recording(
            active,
            cleanup=cleanup,
            discard=discard,
            source=source,
        )

    def toggle_recording(
        self,
        cleanup: bool | None = None,
    ) -> RecordSessionResponse:
        now = self._clock()
        with self._lock:
            cooldown_seconds = max(0, self.settings.toggle_cooldown_ms) / 1000
            if (
                cooldown_seconds
                and self._last_toggle_accepted_at
                and now - self._last_toggle_accepted_at < cooldown_seconds
            ):
                return {
                    "status": "recording" if self._active is not None else "ready",
                    "action": "ignored",
                    "reason": "toggle_cooldown",
                    "cooldown_ms": self.settings.toggle_cooldown_ms,
                }
            self._last_toggle_accepted_at = now
            if self._active is None:
                handle = self._create_recording()
                handle.start()
                self._active = ActiveRecording(handle, now)
                self._status = "recording"
                return {"status": "recording", "action": "started", "already_recording": False}

            active = self._pop_active()

        duration_ms = (self._clock() - active.started_at) * 1000
        over_maximum = (
            self.settings.max_recording_ms > 0
            and duration_ms > self.settings.max_recording_ms
        )
        discard = duration_ms < self.settings.min_recording_ms or over_maximum
        result = self._finish_recording(
            active,
            cleanup=cleanup,
            discard=discard,
            discard_reason="max_recording_duration" if over_maximum else None,
            source="/record/toggle",
        )
        result["status"] = "ready"
        result["action"] = "discarded" if discard else "stopped"
        return result

    def _finish_recording(
        self,
        active: ActiveRecording,
        *,
        cleanup: bool | None,
        discard: bool,
        discard_reason: str | None = None,
        source: str,
    ) -> RecordSessionResponse:
        duration_ms = round((self._clock() - active.started_at) * 1000, 3)
        if discard:
            try:
                active.handle.discard()
                result: RecordSessionResponse = {"status": "ready", "duration_ms": duration_ms, "discarded": True}
                if discard_reason:
                    result["reason"] = discard_reason
                    result["max_recording_ms"] = self.settings.max_recording_ms
                return result
            finally:
                self._set_ready_if_no_active()

        try:
            result = dict(
                active.handle.stop(
                    cleanup=cleanup,
                    source=source,
                    before_transcribe=self._mark_transcribing,
                )
            )
            result["duration_ms"] = duration_ms
            return result
        finally:
            self._set_ready_if_no_active()

    def _pop_active(self) -> ActiveRecording:
        if self._active is None:
            raise RuntimeError("Recorder is not running")
        active = self._active
        self._active = None
        self._status = "stopping"
        return active

    def _mark_transcribing(self) -> None:
        with self._lock:
            if self._status == "stopping":
                self._status = "transcribing"

    def _set_ready_if_no_active(self) -> None:
        with self._lock:
            if self._active is None and self._status in {"stopping", "transcribing"}:
                self._status = "ready"

    def _create_recording(self) -> RecordingHandle:
        if self._streaming is not None:
            return StreamingRecording(
                self._streaming,
                self._streaming.create_recording(),
                self.settings.debug_capture,
            )
        return BatchRecording(self._recorder_factory(self.settings), self._transcribe)
