from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from transclip.daemon.common import CommandResult
from transclip.recording_ops import ToggleOutcome
from transclip.service.types import ServiceHealthResponse
from transclip.settings import Settings


class FetchHealth(Protocol):
    def __call__(self, settings: Settings) -> tuple[ServiceHealthResponse | None, str | None]: ...


class ToggleRecording(Protocol):
    def __call__(self, settings: Settings, *, paste: bool) -> ToggleOutcome: ...


class ServiceAction(Protocol):
    def __call__(self, action: str, **kwargs: Any) -> CommandResult: ...


class ReadHistory(Protocol):
    def __call__(self, *, limit: int) -> list[dict[str, Any]]: ...


class WriteClipboard(Protocol):
    def __call__(self, text: str) -> None: ...


def _default_fetch_health() -> FetchHealth:
    from transclip.service.client_health import fetch_service_health_result

    return fetch_service_health_result


def _default_toggle_recording() -> ToggleRecording:
    from transclip.recording_ops import toggle_recording

    return toggle_recording


def _default_service_action() -> ServiceAction:
    from transclip.daemon import service_action

    return service_action


def _default_read_history() -> ReadHistory:
    from transclip.history import read_history

    return read_history


def _default_write_clipboard() -> WriteClipboard:
    from transclip.desktop.paste import SystemClipboard

    def write(text: str) -> None:
        SystemClipboard().write(text)

    return write


@dataclass(frozen=True, slots=True)
class TrayPorts:
    fetch_health: FetchHealth = field(default_factory=_default_fetch_health)
    toggle_recording: ToggleRecording = field(default_factory=_default_toggle_recording)
    service_action: ServiceAction = field(default_factory=_default_service_action)
    read_history: ReadHistory = field(default_factory=_default_read_history)
    write_clipboard: WriteClipboard = field(default_factory=_default_write_clipboard)


def default_tray_ports() -> TrayPorts:
    return TrayPorts()
