from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from transclip.mode_routing import VoiceMode

# Dictation lifecycle status reported by DictationSession.status() and echoed on
# the /health and /record/* responses.
DictationStatus = Literal["recording", "ready"]
# The action a /record/toggle (or /record/stop) call resolved to.
ToggleAction = Literal["started", "stopped", "discarded", "ignored"]
# Why a recording was discarded / ignored.
DiscardReason = Literal["toggle_cooldown", "max_recording_duration"]
# The originating endpoint recorded for a record/transcribe session.
RecordSource = Literal[
    "/record/stop",
    "/record/toggle",
    "/transcribe",
    "/cleanup/transcribe",
    "/eval/incremental",
]


class ServiceHealthResponse(TypedDict, total=False):
    status: DictationStatus
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
    max_recording_ms: int
    toggle_cooldown_ms: int
    paste_injection_delay_ms: int
    clipboard_restore_delay_ms: int
    restore_clipboard_after_paste: bool
    streaming_partial_supported: bool


class RecordSessionResponse(TypedDict, total=False):
    status: DictationStatus
    action: ToggleAction
    text: str
    duration_ms: float
    discarded: bool
    already_recording: bool
    reason: DiscardReason
    cooldown_ms: int
    max_recording_ms: int
    history_error: str
    log_error: str
    service_url: str
    paste: dict[str, Any]
    timestamp: str
    voice_mode: VoiceMode
    voice_trigger: str
    voice_literal: bool
    shell: dict[str, Any] | None
    timings_ms: dict[str, float]
    debug_capture_dir: str


class CleanupTextResponse(TypedDict, total=False):
    text: str
    backend: str
    timings_ms: dict[str, float]
    voice_mode: VoiceMode
    voice_trigger: str
    voice_literal: bool
    cleanup_backend: str


JsonPayload = dict[str, object]


# /transcribe responses carry every record-session field plus ASR-specific
# detail. Keys are optional (total=False) because the serializer omits cleanup,
# voice_trigger, and debug_capture_dir when the pipeline does not produce them.
class TranscribeResponse(RecordSessionResponse, total=False):
    raw_asr: str
    cleanup: dict[str, Any]
    cleanup_enabled: bool
    submit: bool | None
    asr_backend: str
    asr_model: str
    cleanup_backend: str
