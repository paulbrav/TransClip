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
    cleanup_enabled: bool
    voice_mode_routing_enabled: bool
    voice_model_cleanup_always_on: bool
    voice_mode_shell_enabled: bool
    text_model_runtime: str
    text_model: str
    language: str
    min_recording_ms: int
    toggle_cooldown_ms: int
    clipboard_restore_delay_ms: int
    restore_clipboard_after_paste: bool
    text_delivery_mode: str
    focus_aware_paste: bool
    terminal_wm_class_patterns: str


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
    backend: str
    timings_ms: dict[str, float]
    voice_mode: str
    voice_trigger: str
    voice_literal: bool
    cleanup_backend: str


JsonPayload = dict[str, object]


class TranscribeResponse(RecordSessionResponse):
    raw_asr: str
    cleanup: dict[str, Any]
    cleanup_enabled: bool
    submit: bool | None
