from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError

from transclip.daemon import append_toggle_log
from transclip.desktop.paste import paste_transcript
from transclip.service import InferenceClient
from transclip.service.types import RecordSessionResponse

from .history import timestamp
from .settings import Settings


@dataclass(slots=True)
class ToggleOutcome:
    ok: bool
    payload: RecordSessionResponse
    service_url: str
    error_message: str = ""
    paste_failed_message: str = ""

    @property
    def latest_transcript(self) -> str:
        if self.payload.get("action") == "stopped" and self.payload.get("text"):
            return str(self.payload["text"])
        return ""

    @property
    def notification_message(self) -> str:
        return self.error_message or self.paste_failed_message


def toggle_recording(
    settings: Settings,
    paste: bool = False,
    client: InferenceClient | None = None,
) -> ToggleOutcome:
    client = client or InferenceClient(settings)
    service_url = getattr(client, "base_url", f"http://{settings.host}:{settings.port}")
    try:
        result = client.record_toggle()
    except HTTPError as exc:
        message = f"TransClip service rejected /record/toggle with HTTP {exc.code}."
        _log_toggle_error(message, service_url)
        return ToggleOutcome(False, {}, service_url, error_message=message)
    except URLError:
        message = "TransClip service is not running."
        _log_toggle_error(message, service_url)
        return ToggleOutcome(False, {}, service_url, error_message=message)

    result["service_url"] = service_url
    paste_failed_message = ""
    if paste and result.get("action") == "stopped" and result.get("text"):
        paste_result = paste_transcript(str(result["text"]), settings)
        result["paste"] = asdict(paste_result)
        if not paste_result.pasted:
            detail = f" {paste_result.error_detail}" if paste_result.error_detail else ""
            paste_failed_message = "Paste failed. The transcript is still on the clipboard." + detail
    if "timestamp" not in result:
        result["timestamp"] = timestamp()
    log_error = _append_toggle_log(result)
    if log_error:
        result["log_error"] = log_error
    return ToggleOutcome(True, result, service_url, paste_failed_message=paste_failed_message)


def _log_toggle_error(message: str, service_url: str) -> None:
    _append_toggle_log(
        {
            "timestamp": timestamp(),
            "action": "error",
            "error": message,
            "service_url": service_url,
        }
    )


def _append_toggle_log(event: dict) -> str:
    try:
        append_toggle_log(event)
    except Exception as exc:
        return str(exc)
    return ""
