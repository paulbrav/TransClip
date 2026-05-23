from __future__ import annotations

from typing import Any

from transclip.cleanup import CleanupPlan
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.settings import Settings, active_hotkey, paste_shortcut

_SETTINGS_HEALTH_FIELDS = (
    "cleanup_enabled",
    "voice_mode_routing_enabled",
    "voice_model_cleanup_always_on",
    "voice_mode_shell_enabled",
    "text_model_runtime",
    "text_model",
    "language",
    "max_recording_seconds",
    "min_recording_ms",
    "toggle_cooldown_ms",
    "clipboard_restore_delay_ms",
    "restore_clipboard_after_paste",
)


def settings_health_payload(
    settings: Settings,
    runtime: PlatformRuntime,
) -> dict[str, Any]:
    payload = {field: getattr(settings, field) for field in _SETTINGS_HEALTH_FIELDS}
    payload["hotkey"] = active_hotkey(settings, runtime)
    payload["paste_shortcut"] = paste_shortcut(settings, runtime)
    return payload


def build_health_status(
    *,
    status: str,
    settings: Settings,
    asr_backend_name: str,
    asr_model: str,
    cleanup_backend: str,
    dictation_cleanup: str,
    runtime: PlatformRuntime | None = None,
) -> dict[str, Any]:
    platform_runtime = get_runtime(runtime)
    return {
        "status": status,
        "asr_backend": asr_backend_name,
        "asr_model": asr_model,
        "cleanup_backend": cleanup_backend,
        "dictation_cleanup": dictation_cleanup,
        **settings_health_payload(settings, platform_runtime),
    }


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
