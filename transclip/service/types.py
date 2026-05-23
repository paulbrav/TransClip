from __future__ import annotations

from typing import Any, TypedDict


class ServiceHealthResponse(TypedDict, total=False):
    status: str
    asr_backend: str
    asr_model: str
    cleanup_backend: str
    dictation_cleanup: str
    hotkey: str
    paste_shortcut: str


class RecordSessionResponse(TypedDict, total=False):
    status: str
    action: str
    text: str
    duration_ms: float
    discarded: bool
    already_recording: bool
    reason: str
    cooldown_ms: int
    history_error: str
    log_error: str
    service_url: str
    paste: dict[str, Any]
    timestamp: str
    voice_mode: str
    voice_trigger: str
    voice_literal: bool
    shell: dict[str, Any]
    timings_ms: dict[str, float]
    debug_capture_dir: str


class CleanupTextResponse(TypedDict, total=False):
    text: str
    voice_mode: str
    voice_trigger: str
    voice_literal: bool
    cleanup_backend: str


class TranscribeResponse(RecordSessionResponse):
    raw_asr: str
