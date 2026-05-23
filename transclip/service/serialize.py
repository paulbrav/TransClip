from __future__ import annotations

from dataclasses import asdict

from transclip.cleanup import CleanupResult
from transclip.transcript_pipeline import TranscriptOutcome, shell_metadata

from .types import CleanupTextResponse, TranscribeResponse


def to_cleanup_text_response(result: CleanupResult) -> CleanupTextResponse:
    return {
        "text": result.text,
        "backend": result.backend,
        "timings_ms": dict(result.timings_ms),
    }


def to_transcribe_response(
    outcome: TranscriptOutcome,
    *,
    timings_ms: dict[str, float],
    debug_capture_dir: str | None,
) -> TranscribeResponse:
    response: TranscribeResponse = {
        "text": outcome.text,
        "raw_asr": outcome.raw_asr,
        "voice_mode": outcome.voice_mode,
        "voice_literal": outcome.voice_literal,
        "timings_ms": dict(timings_ms),
        "asr_backend": outcome.asr_backend,
        "asr_model": outcome.asr_model,
        "cleanup_backend": outcome.cleanup_backend,
        "cleanup_enabled": outcome.cleanup_enabled,
        "shell": shell_metadata(outcome.shell),
        "submit": outcome.submit,
    }
    if outcome.voice_trigger is not None:
        response["voice_trigger"] = outcome.voice_trigger
    if outcome.cleanup is not None:
        response["cleanup"] = asdict(outcome.cleanup)
    if debug_capture_dir is not None:
        response["debug_capture_dir"] = debug_capture_dir
    return response
