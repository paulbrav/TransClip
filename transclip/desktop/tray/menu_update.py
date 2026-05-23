from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from transclip.history import history_file_signature

from .menu import tray_action_label, tray_status_label
from .session import TraySession, latest_history_text, preview_text


@dataclass(frozen=True, slots=True)
class TrayMenuSnapshot:
    status_label: str
    toggle_label: str
    latest_enabled: bool
    model_cleanup_label: str
    health_icon: str


@dataclass(slots=True)
class HistoryMenuState:
    signature: object
    refreshing: bool = False


class TrayMenuView(Protocol):
    def set_label(self, ref: str, text: str) -> None: ...

    def set_enabled(self, ref: str, enabled: bool) -> None: ...

    def set_model_labels(self, rows: Sequence[tuple[Any, str]]) -> None: ...

    def rebuild_history(self, entries: Sequence[tuple[str, str]]) -> None: ...

    def set_health_icon(self, icon: str) -> None: ...


def compute_tray_menu_snapshot(session: TraySession) -> TrayMenuSnapshot:
    health = session.health
    return TrayMenuSnapshot(
        status_label=tray_status_label(health.status, health.detail),
        toggle_label=tray_action_label("toggle", recording=health.recording, settings=session.settings),
        latest_enabled=bool(session.latest or latest_history_text()),
        model_cleanup_label=tray_action_label(
            "model_cleanup",
            recording=health.recording,
            settings=session.settings,
        ),
        health_icon=health.icon,
    )


def history_preview_entries(session: TraySession, limit: int = 5) -> tuple[tuple[str, str], ...]:
    return tuple(
        (preview_text(str(event.get("text") or "")), str(event.get("text") or ""))
        for event in session.history_events(limit=limit)
    )


def should_refresh_history(state: HistoryMenuState, signature: object | None, *, force: bool = False) -> bool:
    if state.refreshing:
        return False
    if force:
        return True
    return signature != state.signature


def apply_menu_snapshot(
    snapshot: TrayMenuSnapshot,
    view: TrayMenuView,
    *,
    model_rows: Sequence[tuple[Any, str]],
) -> None:
    view.set_label("status_item", snapshot.status_label)
    view.set_label("toggle_item", snapshot.toggle_label)
    view.set_enabled("latest_item", snapshot.latest_enabled)
    view.set_label("model_cleanup_item", snapshot.model_cleanup_label)
    view.set_model_labels(model_rows)
    view.set_health_icon(snapshot.health_icon)


def after_tray_action(
    action: Callable[[], object],
    *,
    update_menu: Callable[[], None],
    history_state: HistoryMenuState | None = None,
    refresh_history: Callable[[], None] | None = None,
    before_update: Callable[[], None] | None = None,
) -> object:
    outcome = action()
    if (
        history_state is not None
        and refresh_history is not None
        and getattr(outcome, "latest_transcript", None)
    ):
        history_state.signature = object()
        refresh_history()
    if before_update is not None:
        before_update()
    update_menu()
    return outcome


def apply_tray_menu_update(
    session: TraySession,
    view: TrayMenuView,
    *,
    model_items: Sequence[tuple[Any, Any]],
) -> TrayMenuSnapshot:
    from .session import model_menu_label

    snapshot = compute_tray_menu_snapshot(session)
    label_pairs = [
        (item, model_menu_label(row.model_id, row.backend, session.settings))
        for item, row in model_items
    ]
    apply_menu_snapshot(snapshot, view, model_rows=label_pairs)
    return snapshot


def refresh_history_menu(
    session: TraySession,
    state: HistoryMenuState,
    view: TrayMenuView,
    *,
    signature: object | None = None,
    force: bool = False,
) -> None:
    signature = history_file_signature() if signature is None else signature
    if not should_refresh_history(state, signature, force=force):
        return
    state.refreshing = True
    try:
        events = history_preview_entries(session)
        if events:
            view.rebuild_history(events)
        else:
            view.rebuild_history([("No recent transcripts", "")])
        state.signature = signature
    finally:
        state.refreshing = False
