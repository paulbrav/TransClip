from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from transclip.asr import ASRBackend, build_asr_backend
from transclip.audio import AudioRecorder
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
from .session import DictationSession


class InferenceEngine:
    def __init__(
        self,
        settings: Settings,
        asr_backend: ASRBackend | None = None,
        cleanup_backend: CleanupBackend | None = None,
        text_backend: TextGenerationBackend | None = None,
    ):
        self.settings = settings
        self.asr_backend = asr_backend or build_asr_backend(settings)
        self.cleanup_backend = cleanup_backend or FaithfulRuleCleanupBackend()
        self.text_backend = text_backend or build_text_generation_backend(settings)
        self.transcript_processor = TranscriptProcessor(
            settings,
            rule_cleanup=self.cleanup_backend,
            model_cleanup=ModelCleanupProcessor(self.text_backend),
            shell_command=ShellCommandProcessor(settings, self.text_backend),
        )
        self.debug_capture = DebugCapture(settings)
        self.dictation_session = DictationSession(
            settings,
            transcribe=self._transcribe_for_session,
            recorder_factory=lambda current_settings: AudioRecorder(current_settings),
        )

    def health(self) -> dict[str, Any]:
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
            runtime=get_runtime(),
        )

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
    ) -> dict[str, Any]:
        result = self.dictation_session.toggle_recording(cleanup=cleanup)
        return _with_optional_history(
            result,
            self.settings,
            source="/record/toggle",
            record_history=record_history,
            duration_ms=result.get("duration_ms"),
        )

    def cleanup_text(self, text: str) -> dict[str, Any]:
        result = self.transcript_processor.cleanup_dictation(text)
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
        end_to_end_ms = round((perf_counter() - start) * 1000, 3)
        timings_ms = {**outcome.timings_ms, "end_to_end": end_to_end_ms}
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
        result = outcome.to_dict()
        result["timings_ms"] = timings_ms
        result["debug_capture_dir"] = str(capture_dir) if capture_dir else None
        return _with_optional_history(
            result,
            self.settings,
            source=source,
            record_history=record_history,
        )

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


def _with_optional_history(
    result: dict[str, Any],
    settings: Settings,
    *,
    source: str,
    record_history: bool,
    duration_ms: float | None = None,
) -> dict[str, Any]:
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
    result: dict[str, Any],
    settings: Settings,
    source: str,
    duration_ms: float | None = None,
) -> str | None:
    try:
        append_transcript_history(result, settings, source=source, duration_ms=duration_ms)
    except Exception as exc:
        return str(exc)
    return None
