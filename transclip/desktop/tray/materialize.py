from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from transclip.models import ModelRow

from .menu import (
    TrayAction,
    TrayMenuNode,
    asr_model_choices,
    tray_action_label,
    tray_status_label,
    tray_submenu_is_history,
    tray_submenu_title,
)
from .session import TraySession


class TrayMenuSink(Protocol):
    def separator(self) -> None: ...

    def status_label(self, ref: str, text: str) -> None: ...

    def action(
        self,
        ref: str,
        label: str,
        action: TrayAction,
        *,
        enabled: bool = True,
        callback: Callable[..., Any] | None = None,
    ) -> None: ...

    def history_submenu(self, ref: str, title: str, on_open: Callable[[], None] | None = None) -> None: ...

    def model_submenu(self, ref: str, title: str, choices: Iterable[tuple[str, ModelRow]]) -> None: ...


def materialize_tray_menu(
    nodes: Iterable[TrayMenuNode],
    session: TraySession,
    sink: TrayMenuSink,
    *,
    action_callbacks: dict[TrayAction, Callable[..., Any]],
    initial_status_label: bool = False,
    on_history_open: Callable[[], None] | None = None,
) -> None:
    for node in nodes:
        if node.kind == "separator":
            sink.separator()
            continue
        if node.kind == "label":
            health = session.health
            sink.status_label(
                node.ref,
                tray_status_label(health.status, health.detail, initial=initial_status_label),
            )
            continue
        if node.kind == "submenu":
            assert node.action is not None
            title = tray_submenu_title(node.action)
            if tray_submenu_is_history(node.action):
                sink.history_submenu(node.ref, title, on_open=on_history_open)
                continue
            sink.model_submenu(
                node.ref,
                title,
                asr_model_choices(session.settings, session.runtime),
            )
            continue
        assert node.action is not None
        label = tray_action_label(
            node.action,
            recording=session.health.recording,
            settings=session.settings,
        )
        enabled = node.action != "copy_latest" or bool(session.latest)
        sink.action(
            node.ref,
            label,
            node.action,
            enabled=enabled,
            callback=action_callbacks[node.action],
        )
