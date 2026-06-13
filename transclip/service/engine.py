from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from transclip.asr import (
    GRANITE_NAR_BUCKET_SECONDS,
    MLX_SHORT_AUDIO_BUCKET_SECONDS,
    MLX_WARM_BUCKET_MAX_SECONDS,
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


class StopSignal(Protocol):
    def wait(self, timeout: float) -> bool: ...

    def is_set(self) -> bool: ...


class WaveformTranscriber(Protocol):
    def __call__(self, waveform: Any, sample_rate: int = 16000) -> TranscriptionResult: ...


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
        if warm_asr:
            # Warmup failure (e.g. weights not yet downloaded) must not abort
            # startup: serve degraded and surface the error per-request, as the
            # lazy-loading path always did.
            try:
                self.warm_asr()
            except Exception:
                logging.getLogger(__name__).exception(
                    "ASR warmup failed; continuing with lazy model load"
                )
        self._streaming = streaming if streaming is not None else self._build_incremental_adapter()
        self.dictation_session = DictationSession(
            settings,
            transcribe=self._transcribe_for_session,
            recorder_factory=lambda current_settings: AudioRecorder(current_settings),
            streaming=self._streaming,
        )

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
        source: str = "/record/stop",
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
        with tempfile.TemporaryDirectory(prefix="transclip-warmup-") as tmp:
            tmp_path = Path(tmp)
            warm_seconds = _asr_warm_seconds(self.asr_backend)
            speech_warmup = _mlx_speech_warmup_wav(tmp_path, sample_rate) if _mlx_settings(self.settings) else None
            for seconds in warm_seconds:
                pcm16_warmup = _warmup_pcm16_chirp(sample_rate, seconds=seconds)
                wav_path = write_wav(
                    tmp_path / f"warmup-{seconds:g}s.wav",
                    pcm16_warmup,
                    sample_rate,
                )
                self.asr_backend.transcribe(wav_path, keywords=[])
            if speech_warmup is not None:
                self.asr_backend.transcribe(speech_warmup, keywords=[])

    def warm_bucket_shapes(self, stop_event: StopSignal) -> None:
        """Compile remaining backend input buckets in the background after readiness."""
        if self.asr_backend.name == "mlx-audio":
            # Long MLX background warmup regresses short interactive dictation:
            # it can churn the compiled state and make the next real mic clip
            # pay a 20s+ first-pass cost. Keep MLX warmup in startup only.
            return

        transcribe_waveform = _waveform_transcriber(self.asr_backend)
        max_seconds = max(0, int(self.settings.warm_bucket_shapes_s))
        if transcribe_waveform is None or max_seconds <= 0:
            return

        import numpy as np

        logger = logging.getLogger(__name__)
        sample_rate = max(1, self.settings.sample_rate)
        for seconds in _bucket_warm_seconds(max_seconds):
            while _dictation_busy(self.dictation_session.status()):
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
        source: str,
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


def _asr_warm_seconds(backend: ASRBackend) -> list[float]:
    if backend.name == "mlx-audio":
        bucket_seconds = max(1, int(MLX_SHORT_AUDIO_BUCKET_SECONDS))
        return [
            float(seconds)
            for seconds in range(bucket_seconds, MLX_WARM_BUCKET_MAX_SECONDS + 1, bucket_seconds)
        ]
    return [0.25]


def _mlx_settings(settings: Settings) -> bool:
    return "mlx" in settings.asr_backend


def _mlx_speech_warmup_wav(directory: Path, sample_rate: int) -> Path | None:
    runtime = get_runtime()
    if runtime.system() != "Darwin":
        return None
    if not runtime.which("say") or not runtime.which("afconvert"):
        return None

    data_format = f"LEI16@{sample_rate}"
    caf_path = directory / "speech-warmup.caf"
    wav_path = directory / "speech-warmup.wav"
    try:
        runtime.run(
            [
                "say",
                "-o",
                str(caf_path),
                f"--data-format={data_format}",
                "Testing 1, 2, 3.",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        runtime.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                data_format,
                str(caf_path),
                str(wav_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        logging.getLogger(__name__).debug(
            "Speech ASR warmup generation failed; using chirp fallback",
            exc_info=True,
        )
        return None
    return wav_path if wav_path.exists() else None


def _dictation_busy(status: str) -> bool:
    return status in {"recording", "stopping", "transcribing"}


def _warmup_pcm16_chirp(sample_rate: int, seconds: float = 0.25) -> bytes:
    frame_count = max(1, int(sample_rate * seconds))
    peak = 0.08 * 32767.0
    frames = bytearray()
    for index in range(frame_count):
        progress = index / max(1, frame_count - 1)
        frequency = 180.0 + 420.0 * progress
        sample = int(peak * math.sin(2.0 * math.pi * frequency * index / sample_rate))
        frames.extend(sample.to_bytes(2, "little", signed=True))
    return bytes(frames)


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
