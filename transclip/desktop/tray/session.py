from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from transclip.models import model_display_name
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.recording_ops import ToggleOutcome
from transclip.settings import Settings, patch_settings, settings_path, write_default_settings

from .ports import TrayPorts, default_tray_ports

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
        ports: TrayPorts | None = None,
    ):
        self.settings = settings
        self.explicit_settings_path = explicit_settings_path
        self.runtime = runtime or get_runtime()
        self.ports = ports or default_tray_ports()
        self.latest = ""
        self.health = TrayHealth(status="starting", recording=False, detail="", icon="ready")

    def _update_health(self, **changes: Any) -> None:
        self.health = replace(self.health, **changes)

    def set_health_offline(self, error: str) -> None:
        self._update_health(
            status="offline",
            recording=False,
            detail=f"Service: offline ({error})",
            icon="offline",
        )

    def set_health_status(
        self,
        *,
        status: str,
        recording: bool,
        detail: str,
        icon: TrayIcon,
    ) -> None:
        self._update_health(status=status, recording=recording, detail=detail, icon=icon)

    def refresh_health(self) -> TrayHealth:
        health, error = self.ports.fetch_health(self.settings)
        if error is not None:
            self.set_health_offline(error)
            return self.health
        status = str((health or {}).get("status", "unknown"))
        recording = status == "recording"
        self.set_health_status(
            status=status,
            recording=recording,
            detail=f"Service: {status}",
            icon="recording" if recording else "ready",
        )
        return self.health

    def toggle_record(self) -> ToggleOutcome:
        outcome = self.ports.toggle_recording(self.settings, paste=True)
        if not outcome.ok:
            self._update_health(detail=f"Toggle failed: {outcome.error_message}")
            return outcome
        if outcome.latest_transcript:
            self.latest = outcome.latest_transcript
        self.refresh_health()
        if outcome.payload.get("action") == "started":
            self.set_health_status(
                status="recording",
                recording=True,
                detail="Service: recording",
                icon="recording",
            )
        elif outcome.payload.get("action") == "stopped" and not outcome.paste_failed_message:
            self.set_health_status(
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
            self.ports.write_clipboard(text)
            self._update_health(detail="Copied latest transcript")
        except Exception as exc:
            self._update_health(detail=f"Copy failed: {exc}")
        return self.health.detail

    def copy_latest(self) -> str:
        return self.copy_text(self.latest or latest_history_text(self))

    def start_service(self) -> str:
        result = self.ports.service_action("start")
        self.refresh_health()
        self._update_health(detail=result.detail)
        return self.health.detail

    def restart_service(self) -> str:
        result = self.ports.service_action("restart")
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
            restart = self.ports.service_action("restart")
            label = model_display_name(model_id)
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
            restart = self.ports.service_action("restart")
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

    def history_events(self, limit: int = 5) -> list[dict[str, Any]]:
        return self.ports.read_history(limit=limit)


def latest_history_text(session: TraySession | None = None) -> str:
    if session is not None:
        events = session.history_events(limit=1)
    else:
        from transclip.history import read_history

        events = read_history(limit=1)
    return str(events[0].get("text") or "") if events else ""


def can_copy_latest(session: TraySession, *, cached_history_text: str | None = None) -> bool:
    if session.latest:
        return True
    if cached_history_text is not None:
        return bool(cached_history_text)
    return bool(latest_history_text(session))


def preview_text(text: str, limit: int = 48) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "..."
