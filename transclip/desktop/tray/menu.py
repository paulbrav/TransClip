from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from transclip.models import ModelRow, asr_model_rows
from transclip.platform.runtime import PlatformRuntime
from transclip.settings import Settings

from .session import model_cleanup_label, model_menu_label

TrayAction = Literal[
    "status",
    "toggle",
    "copy_latest",
    "history",
    "start_service",
    "restart_service",
    "model_cleanup",
    "asr_model",
    "set_hotkey",
    "copy_hotkey_setup",
    "open_settings",
    "quit",
]

TrayMenuKind = Literal["label", "separator", "action", "submenu"]

TRAY_ACTION_LABELS: dict[TrayAction, str] = {
    "copy_latest": "Copy latest transcript",
    "history": "Recent transcripts",
    "start_service": "Start service",
    "restart_service": "Restart service",
    "model_cleanup": "Model cleanup always on",
    "asr_model": "ASR model",
    "set_hotkey": "Set hotkey...",
    "copy_hotkey_setup": "Copy hotkey setup command",
    "open_settings": "Open settings",
    "quit": "Quit tray",
}

MACOS_SELECTORS: dict[TrayAction, str] = {
    "toggle": "toggleRecord:",
    "copy_latest": "copyLatest:",
    "start_service": "startService:",
    "restart_service": "restartService:",
    "model_cleanup": "toggleModelCleanup:",
    "copy_hotkey_setup": "copyHotkeySetup:",
    "open_settings": "openSettings:",
    "quit": "quitTray:",
}

MODEL_ITEMS_REF = "model_items"


@dataclass(frozen=True, slots=True)
class TrayMenuNode:
    kind: TrayMenuKind
    action: TrayAction | None = None
    ref: str = ""


def tray_menu_nodes(system: str) -> tuple[TrayMenuNode, ...]:
    hotkey: TrayAction = "copy_hotkey_setup" if system == "Darwin" else "set_hotkey"
    return (
        TrayMenuNode("label", ref="status_item"),
        TrayMenuNode("separator"),
        TrayMenuNode("action", "toggle", "toggle_item"),
        TrayMenuNode("action", "copy_latest", "latest_item"),
        TrayMenuNode("separator"),
        TrayMenuNode("submenu", "history", "history_menu"),
        TrayMenuNode("separator"),
        TrayMenuNode("action", "start_service"),
        TrayMenuNode("action", "restart_service"),
        TrayMenuNode("action", "model_cleanup", "model_cleanup_item"),
        TrayMenuNode("submenu", "asr_model", "model_menu"),
        TrayMenuNode("action", hotkey),
        TrayMenuNode("action", "open_settings"),
        TrayMenuNode("separator"),
        TrayMenuNode("action", "quit"),
    )


def toggle_record_label(recording: bool) -> str:
    return "Stop + paste" if recording else "Record"


def tray_status_label(status: str, detail: str, *, initial: bool = False) -> str:
    if detail:
        return detail
    if initial:
        return "Service: starting"
    return f"Service: {status}"


def tray_action_label(
    action: TrayAction,
    *,
    recording: bool,
    settings: Settings,
) -> str:
    if action == "toggle":
        return toggle_record_label(recording)
    if action == "model_cleanup":
        return model_cleanup_label(settings)
    return TRAY_ACTION_LABELS[action]


def tray_icon_for_health(icon: str, *, system: str) -> str:
    if system == "Darwin":
        return {"recording": "●", "ready": "🎙", "offline": "⚠"}[icon]
    return {
        "recording": "media-record-symbolic",
        "ready": "audio-input-microphone-symbolic",
        "offline": "dialog-warning-symbolic",
    }[icon]


def asr_model_choices(
    settings: Settings,
    runtime: PlatformRuntime | None = None,
) -> Iterator[tuple[str, ModelRow]]:
    for row in asr_model_rows(settings, runtime):
        yield model_menu_label(row.model_id, row.backend, settings), row


def tray_submenu_title(action: TrayAction) -> str:
    if action not in {"history", "asr_model"}:
        raise ValueError(f"not a submenu action: {action!r}")
    return TRAY_ACTION_LABELS[action]


def tray_submenu_is_history(action: TrayAction | None) -> bool:
    return action == "history"
