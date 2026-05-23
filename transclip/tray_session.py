from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .client import InferenceClient
from .daemon_lifecycle import service_action
from .history import read_history
from .hotkey_setup import macos_hotkey_setup_message
from .paste import SystemClipboard
from .platform_runtime import PlatformRuntime, get_runtime
from .recording_ops import ToggleOutcome, toggle_recording
from .settings import Settings, patch_settings, settings_path, write_default_settings

TrayIcon = Literal["recording", "ready", "offline"]


@dataclass(slots=True)
class TrayHealth:
    status: str
    recording: bool
    detail: str
    icon: TrayIcon


class TraySession:
    def __init__(
        self,
        settings: Settings,
        explicit_settings_path: Path | None = None,
        runtime: PlatformRuntime | None = None,
    ):
        self.settings = settings
        self.explicit_settings_path = explicit_settings_path
        self.runtime = runtime or get_runtime()
        self.latest = ""
        self.health = TrayHealth(status="starting", recording=False, detail="", icon="ready")

    def _update_health(self, **changes: Any) -> None:
        self.health = replace(self.health, **changes)

    def refresh_health(self) -> TrayHealth:
        try:
            health = InferenceClient(self.settings).health()
            status = str(health.get("status", "unknown"))
            recording = status == "recording"
            self._update_health(
                status=status,
                recording=recording,
                detail=f"Service: {status}",
                icon="recording" if recording else "ready",
            )
        except Exception as exc:
            self._update_health(
                status="offline",
                recording=False,
                detail=f"Service: offline ({exc})",
                icon="offline",
            )
        return self.health

    def toggle_record(self) -> ToggleOutcome:
        outcome = toggle_recording(self.settings, paste=True)
        if not outcome.ok:
            self._update_health(detail=f"Toggle failed: {outcome.error_message}")
            return outcome
        if outcome.latest_transcript:
            self.latest = outcome.latest_transcript
        self.refresh_health()
        if outcome.payload.get("action") == "started":
            self._update_health(
                status="recording",
                recording=True,
                detail="Service: recording",
                icon="recording",
            )
        elif outcome.payload.get("action") == "stopped" and not outcome.paste_failed_message:
            self._update_health(
                status="ready",
                recording=False,
                detail="Service: ready",
                icon="ready",
            )
        if outcome.paste_failed_message:
            self._update_health(detail=outcome.paste_failed_message)
        return outcome

    def copy_text(self, text: str) -> str:
        if not text:
            self._update_health(detail="No transcript available")
            return self.health.detail
        try:
            SystemClipboard().write(text)
            self._update_health(detail="Copied latest transcript")
        except Exception as exc:
            self._update_health(detail=f"Copy failed: {exc}")
        return self.health.detail

    def copy_latest(self) -> str:
        return self.copy_text(self.latest or latest_history_text())

    def start_service(self) -> str:
        result = service_action("start")
        self.refresh_health()
        self._update_health(detail=result.detail)
        return self.health.detail

    def restart_service(self) -> str:
        result = service_action("restart")
        self.refresh_health()
        self._update_health(detail=result.detail)
        return self.health.detail

    def set_asr_model(self, model_id: str, backend: str) -> str:
        path = self.explicit_settings_path or settings_path()
        try:
            self.settings = patch_settings(
                path,
                asr_model=model_id,
                asr_backend=backend,
            )
            restart = service_action("restart")
            label = model_label(model_id, backend)
            detail = f"ASR model set to {label}; {restart.detail}"
        except Exception as exc:
            detail = f"ASR model update failed: {exc}"
        self._update_health(detail=detail)
        return detail

    def toggle_model_cleanup(self) -> str:
        path = self.explicit_settings_path or settings_path()
        try:
            next_value = not self.settings.voice_model_cleanup_always_on
            self.settings = patch_settings(
                path,
                voice_model_cleanup_always_on=next_value,
            )
            restart = service_action("restart")
            detail = f"Model cleanup always {'on' if next_value else 'off'}; {restart.detail}"
        except Exception as exc:
            detail = f"Model cleanup toggle failed: {exc}"
        self._update_health(detail=detail)
        return detail

    def set_detail(self, detail: str) -> None:
        self._update_health(detail=detail)

    def open_settings(self) -> Path:
        path = self.explicit_settings_path or settings_path()
        write_default_settings(path)
        return path

    def hotkey_setup_message(self) -> str:
        return macos_hotkey_setup_message(
            self.settings,
            self.explicit_settings_path,
            self.runtime,
        )

    def history_events(self, limit: int = 5) -> list[dict[str, Any]]:
        return read_history(limit=limit)


def latest_history_text() -> str:
    events = read_history(limit=1)
    return str(events[0].get("text") or "") if events else ""


def preview_text(text: str, limit: int = 48) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "..."


def model_cleanup_label(settings: Settings) -> str:
    prefix = "✓ " if settings.voice_model_cleanup_always_on else ""
    return prefix + "Model cleanup always on"


def model_menu_label(model_id: str, backend: str, settings: Settings) -> str:
    prefix = "✓ " if settings.asr_model == model_id and settings.asr_backend == backend else ""
    return prefix + model_label(model_id, backend)


def model_label(model_id: str, backend: str) -> str:
    if backend == "granite_nar":
        return "Fast local ASR - Granite 4.1 NAR"
    if backend == "granite":
        if model_id.endswith("-plus"):
            return "Speaker/timestamp ASR - Granite 4.1 Plus"
        return "Keyword-biased ASR - Granite 4.1"
    if backend == "mlx_audio_whisper":
        return "MLX Whisper Turbo"
    if backend == "granite_mlx":
        return "MLX Granite Speech"
    return model_id
