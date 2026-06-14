from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol

from transclip.asr import (
    GRANITE_NAR_BUCKET_SECONDS,
    ASRBackend,
    TranscriptionResult,
    build_asr_backend,
)
from transclip.asr_incremental import IncrementalNarSession, incremental_transcription_enabled
from transclip.audio import AudioRecorder, write_wav
from transclip.best_effort import best_effort
from transclip.cleanup import (
    CleanupBackend,
    FaithfulRuleCleanupBackend,
    ModelCleanupProcessor,
)
from transclip.debug_capture import DebugCapture
from transclip.history import append_transcript_history
from transclip.keyword_restore import restore_keywords
from transclip.mode_routing import route_voice_mode
from transclip.platform.runtime import get_runtime
from transclip.settings import Settings
from transclip.shell_command import ShellCommandProcessor
from transclip.text_generation import TextGenerationBackend, build_text_generation_backend
from transclip.transcript_pipeline import TranscriptProcessor, shell_metadata

from .health import build_health_status, cleanup_labels
from .serialize import to_cleanup_text_response, to_transcribe_response
from .session import DictationSession
from .streaming import StreamingDictationAdapter
from .types import (
    CleanupTextResponse,
    RecordSessionResponse,
    ServiceHealthResponse,
    TranscribeResponse,
)

if TYPE_CHECKING:
    from .types import RecordSource


class StopSignal(Protocol):
    def wait(self, timeout: float) -> bool: ...

    def is_set(self) -> bool: ...


class WaveformTranscriber(Protocol):
    def __call__(self, waveform: Any, sample_rate: int = 16000) -> TranscriptionResult: ...


def _ml_stack_importable() -> bool:
    """Whether the core ML stack imports. Used to classify a warmup failure: a
    pruned torch is an env-broken (operator-fix) condition regardless of how the
    failure surfaced -- ModuleNotFoundError on the NAR path, or a wrapped
    RuntimeError from the GPU device probe (which runs torch in a subprocess) on
    the AR + cuda path -- so the exception type alone is not reliable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _redact_home(message: str | None) -> str | None:
    """Strip the user's home path from wire-exposed error text (the full text
    stays in the journal log). /readyz is loopback + same-origin guarded, but
    there is no reason to put an absolute home path / username on the wire."""
    if message is None:
        return None
    home = str(Path.home())
    return message.replace(home, "~") if home else message


class InferenceEngine:
    def __init__(
        self,
        settings: Settings,
        asr_backend: ASRBackend | None = None,
        cleanup_backend: CleanupBackend | None = None,
        text_backend: TextGenerationBackend | None = None,
        streaming: StreamingDictationAdapter | None = None,
        warm_asr: bool = False,
    ):
        self.settings = settings
        self.cleanup_backend = cleanup_backend or FaithfulRuleCleanupBackend()
        self.text_backend = text_backend or build_text_generation_backend(settings)
        self.transcript_processor = TranscriptProcessor(
            settings,
            rule_cleanup=self.cleanup_backend,
            model_cleanup=ModelCleanupProcessor(self.text_backend),
            shell_command=ShellCommandProcessor(settings, self.text_backend),
        )
        self.debug_capture = DebugCapture(settings)
        self.asr_backend = asr_backend or build_asr_backend(settings)
        # Readiness reflects whether the ASR stack is usable. Default True so
        # lazy-load CLI paths (no warmup) and successful warmups report ready;
        # a warm_asr() failure flips it False and records why (surfaced by /readyz).
        self.asr_ready: bool = True
        self.asr_last_error: str | None = None
        # True only when warmup failed because the ML stack (torch/transformers)
        # could not be imported -- an env-broken condition the operator must fix.
        self.asr_env_broken: bool = False
        if warm_asr:
            # Warmup failure (e.g. weights not yet downloaded) must not abort
            # startup: serve degraded and surface the error per-request, as the
            # lazy-loading path always did. But record it so /readyz reports 503
            # instead of silently 500-ing every transcription.
            try:
                self.warm_asr()
                self.asr_ready = True
                self.asr_last_error = None
            except Exception as exc:
                self.asr_ready = False
                self.asr_last_error = f"{type(exc).__name__}: {exc}"
                # Classify by probing the real condition, not the exception type:
                # a pruned torch surfaces as ModuleNotFoundError on the NAR path
                # but as a wrapped RuntimeError on the AR + cuda path, so the type
                # is unreliable. env_broken => the operator must reinstall the env.
                self.asr_env_broken = not _ml_stack_importable()
                if self.asr_env_broken:
                    logging.getLogger(__name__).error(
                        "ASR warmup failed: ML stack not importable (%s). Reinstall "
                        "the serving env via scripts/setup_gfx1151_env.sh; service is "
                        "degraded and /record/* will fail until torch is restored.",
                        self.asr_last_error,
                    )
                else:
                    logging.getLogger(__name__).exception("ASR warmup failed; continuing with lazy model load")
        self._streaming = streaming if streaming is not None else self._build_incremental_adapter()
        self.dictation_session = DictationSession(
            settings,
            transcribe=self._transcribe_for_session,
            recorder_factory=lambda current_settings: AudioRecorder(current_settings),
            streaming=self._streaming,
        )

    def asr_readiness(self) -> dict[str, object]:
        """Report whether the ASR backend loaded. Drives GET /readyz (200/503)."""
        return {
            "ready": self.asr_ready,
            "env_broken": self.asr_env_broken,
            "asr_backend": self.asr_backend.name,
            "asr_model": self.asr_backend.model,
            "error": _redact_home(self.asr_last_error),
        }

    def health(self) -> ServiceHealthResponse:
        status = self.dictation_session.status()
        cleanup_backend, dictation_cleanup = cleanup_labels(
            self.settings,
            rule_name=self.cleanup_backend.name,
            text_backend=self.text_backend.name,
            text_model=self.text_backend.model_name,
        )
        return build_health_status(
            status=status,
            settings=self.settings,
            asr_backend_name=self.asr_backend.name,
            asr_model=self.asr_backend.model,
            cleanup_backend=cleanup_backend,
            dictation_cleanup=dictation_cleanup,
            streaming_partial_supported=self._streaming is not None,
            runtime=get_runtime(),
        )

    def start_recording(self) -> RecordSessionResponse:
        return self.dictation_session.start_recording()

    def stop_recording(
        self,
        cleanup: bool | None = None,
        discard: bool = False,
        source: RecordSource = "/record/stop",
        record_history: bool = False,
    ) -> RecordSessionResponse:
        result = self.dictation_session.stop_recording(
            cleanup=cleanup,
            discard=discard,
            source=source,
        )
        return _with_optional_history(
            result,
            self.settings,
            source=source,
            record_history=record_history,
            duration_ms=result.get("duration_ms"),
        )

    def toggle_recording(
        self,
        cleanup: bool | None = None,
        record_history: bool = False,
    ) -> RecordSessionResponse:
        result = self.dictation_session.toggle_recording(cleanup=cleanup)
        return _with_optional_history(
            result,
            self.settings,
            source="/record/toggle",
            record_history=record_history,
            duration_ms=result.get("duration_ms"),
        )

    def record_partial(self) -> dict[str, object]:
        partial = self.dictation_session.partial_text()
        status = self.dictation_session.status()
        payload: dict[str, object] = {
            "status": status,
            "partial_text": partial.text,
        }
        if partial.language:
            payload["language"] = partial.language
        return payload

    def cleanup_text(self, text: str) -> CleanupTextResponse:
        result = self.transcript_processor.cleanup_dictation(text)
        return to_cleanup_text_response(result)

    def transcribe(
        self,
        wav_path: Path,
        cleanup: bool | None = None,
        source: str = "/transcribe",
        record_history: bool = False,
        keywords: list[str] | None = None,
    ) -> TranscribeResponse:
        start = perf_counter()
        asr_result = self.asr_backend.transcribe(wav_path, keywords=keywords)
        result = self.process_asr_result(
            asr_result,
            cleanup=cleanup,
            source=source,
            keywords=keywords,
            start_time=start,
            wav_path=wav_path,
        )
        return _with_optional_history(
            result,
            self.settings,
            source=source,
            record_history=record_history,
        )

    def warm_asr(self) -> None:
        """Load and compile the ASR backend before the service reports ready."""
        sample_rate = max(1, self.settings.sample_rate)
        pcm16_silence = b"\x00\x00" * (sample_rate * 2)
        with tempfile.TemporaryDirectory(prefix="transclip-warmup-") as tmp:
            wav_path = write_wav(Path(tmp) / "warmup.wav", pcm16_silence, sample_rate)
            self.asr_backend.transcribe(wav_path, keywords=[])

    def warm_bucket_shapes(self, stop_event: StopSignal) -> None:
        """Compile remaining NAR bucket shapes in the background after readiness."""
        transcribe_waveform = _waveform_transcriber(self.asr_backend)
        max_seconds = max(0, int(self.settings.warm_bucket_shapes_s))
        if transcribe_waveform is None or max_seconds <= 0:
            return

        import numpy as np

        logger = logging.getLogger(__name__)
        sample_rate = max(1, self.settings.sample_rate)
        for seconds in _bucket_warm_seconds(max_seconds):
            while self.dictation_session.status() == "recording":
                if stop_event.wait(1.0):
                    return
            if stop_event.is_set():
                return
            try:
                transcribe_waveform(
                    np.zeros(seconds * sample_rate, dtype=np.float32),
                    sample_rate=sample_rate,
                )
                logger.info("Pre-warmed ASR bucket shape at %ss", seconds)
            except Exception:
                logger.exception("Bucket pre-warm failed at %ss; aborting pre-warm", seconds)
                return

    def process_asr_result(
        self,
        asr_result: TranscriptionResult,
        *,
        cleanup: bool | None,
        source: str,
        keywords: list[str] | None = None,
        end_to_end_ms: float | None = None,
        start_time: float | None = None,
        wav_path: Path | None = None,
    ) -> TranscribeResponse:
        # end_to_end must span ASR plus all post-processing (keyword restore,
        # routing, cleanup); callers pass start_time taken before the ASR pass.
        start = start_time if start_time is not None else perf_counter()
        raw_asr = restore_keywords(asr_result.text, keywords or [])
        route = route_voice_mode(
            raw_asr,
            routing_enabled=self.settings.voice_mode_routing_enabled,
            shell_enabled=self.settings.voice_mode_shell_enabled,
        )
        outcome = self.transcript_processor.process(
            raw_asr,
            route,
            cleanup=cleanup,
            asr_backend=asr_result.backend,
            asr_model=asr_result.model,
            timings_ms=dict(asr_result.timings_ms),
        )
        if end_to_end_ms is None:
            end_to_end_ms = round((perf_counter() - start) * 1000, 3)
        timings_ms = {**outcome.timings_ms, "end_to_end": end_to_end_ms}
        capture_dir = None
        if wav_path is not None:
            capture_dir = self.debug_capture.write(
                wav_path=wav_path,
                raw_asr=asr_result.text,
                cleaned=outcome.text,
                timings=timings_ms,
                model_versions={
                    "asr_backend": asr_result.backend,
                    "asr_model": asr_result.model,
                    "cleanup_backend": outcome.cleanup_backend,
                    "text_model_runtime": self.settings.text_model_runtime,
                    "text_model": self.settings.text_model,
                },
                metadata={
                    "voice_mode": outcome.voice_mode,
                    "voice_trigger": outcome.voice_trigger,
                    "voice_literal": outcome.voice_literal,
                    "shell": shell_metadata(outcome.shell),
                },
            )
        return to_transcribe_response(
            outcome,
            timings_ms=timings_ms,
            debug_capture_dir=str(capture_dir) if capture_dir else None,
        )

    def _transcribe_for_session(
        self,
        wav_path: Path,
        cleanup: bool | None,
        source: RecordSource,
    ) -> TranscribeResponse:
        return self.transcribe(
            wav_path,
            cleanup=cleanup,
            source=source,
            record_history=False,
        )

    def _build_incremental_adapter(self) -> StreamingDictationAdapter | None:
        if not incremental_transcription_enabled(self.settings):
            return None
        transcribe_waveform = _waveform_transcriber(self.asr_backend)
        if transcribe_waveform is None:
            return None
        settings = self.settings
        backend = self.asr_backend

        def transcribe_chunk(waveform: object) -> TranscriptionResult:
            # The batch path resamples in TorchAudioPreparer; mirror that here
            # so non-16kHz capture rates do not feed the model raw.
            return backend.transcribe_waveform(waveform, sample_rate=settings.sample_rate)

        def session_factory() -> IncrementalNarSession:
            return IncrementalNarSession(
                transcribe_chunk,
                sample_rate=settings.sample_rate,
                commit_threshold_s=settings.incremental_commit_threshold_s,
                backend_name=backend.name,
                model_name=backend.model,
            )

        return StreamingDictationAdapter(settings, session_factory, self.process_asr_result)


def _waveform_transcriber(backend: ASRBackend) -> WaveformTranscriber | None:
    transcribe_waveform = getattr(backend, "transcribe_waveform", None)
    if not callable(transcribe_waveform):
        return None
    return transcribe_waveform


def _bucket_warm_seconds(max_seconds: int) -> range:
    bucket_step_s = max(1, int(GRANITE_NAR_BUCKET_SECONDS))
    return range(bucket_step_s * 2, max_seconds + 1, bucket_step_s)


def _with_optional_history(
    result: RecordSessionResponse | TranscribeResponse,
    settings: Settings,
    *,
    source: str,
    record_history: bool,
    duration_ms: float | None = None,
) -> RecordSessionResponse | TranscribeResponse:
    if not record_history:
        return result
    history_error = _append_transcript_history(
        result,
        settings,
        source=source,
        duration_ms=duration_ms,
    )
    if history_error:
        result["history_error"] = history_error
    return result


def _append_transcript_history(
    result: RecordSessionResponse | TranscribeResponse,
    settings: Settings,
    source: str,
    duration_ms: float | None = None,
) -> str | None:
    return best_effort(
        lambda: append_transcript_history(
            result,
            settings,
            source=source,
            duration_ms=duration_ms,
        )
    )
