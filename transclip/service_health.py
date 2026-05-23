from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from transclip.platform.runtime import PlatformRuntime, get_runtime

from .cleanup import CleanupPlan
from .settings import Settings, active_hotkey, paste_shortcut


@dataclass(slots=True)
class HealthStatus:
    status: str
    asr_backend: str
    asr_model: str
    cleanup_backend: str
    dictation_cleanup: str
    cleanup_enabled: bool
    voice_mode_routing_enabled: bool
    voice_model_cleanup_always_on: bool
    voice_mode_shell_enabled: bool
    text_model_runtime: str
    text_model: str
    language: str
    max_recording_seconds: int
    min_recording_ms: int
    toggle_cooldown_ms: int
    hotkey: str
    paste_shortcut: str
    clipboard_restore_delay_ms: int
    restore_clipboard_after_paste: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_health_status(
    *,
    status: str,
    settings: Settings,
    asr_backend_name: str,
    asr_model: str,
    cleanup_backend: str,
    dictation_cleanup: str,
    runtime: PlatformRuntime | None = None,
) -> HealthStatus:
    platform_runtime = get_runtime(runtime)
    return HealthStatus(
        status=status,
        asr_backend=asr_backend_name,
        asr_model=asr_model,
        cleanup_backend=cleanup_backend,
        dictation_cleanup=dictation_cleanup,
        cleanup_enabled=settings.cleanup_enabled,
        voice_mode_routing_enabled=settings.voice_mode_routing_enabled,
        voice_model_cleanup_always_on=settings.voice_model_cleanup_always_on,
        voice_mode_shell_enabled=settings.voice_mode_shell_enabled,
        text_model_runtime=settings.text_model_runtime,
        text_model=settings.text_model,
        language=settings.language,
        max_recording_seconds=settings.max_recording_seconds,
        min_recording_ms=settings.min_recording_ms,
        toggle_cooldown_ms=settings.toggle_cooldown_ms,
        hotkey=active_hotkey(settings, platform_runtime),
        paste_shortcut=paste_shortcut(settings, platform_runtime),
        clipboard_restore_delay_ms=settings.clipboard_restore_delay_ms,
        restore_clipboard_after_paste=settings.restore_clipboard_after_paste,
    )


def cleanup_labels(
    settings: Settings,
    *,
    rule_name: str,
    text_backend: str,
    text_model: str,
) -> tuple[str, str]:
    plan = CleanupPlan.from_settings(settings)
    return plan.backend_label(
        rule_name=rule_name,
        text_backend=text_backend,
        text_model=text_model,
    ), plan.dictation_mode
