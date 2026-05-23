from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from transclip.platform.runtime import open_path

from .menu import MODEL_ITEMS_REF
from .menu_update import (
    HistoryMenuState,
    TrayMenuView,
    after_tray_action,
    apply_tray_menu_update,
    refresh_history_menu,
)
from .session import TraySession


class TrayActionHost(Protocol):
    def toggle_record(self) -> object: ...

    def copy_latest(self) -> None: ...

    def run_tray_action(
        self,
        action: Callable[[], object],
        *,
        before_update: Callable[[], None] | None = None,
    ) -> object: ...

    def update_menu(self) -> None: ...


class TrayController:
    def __init__(
        self,
        session: TraySession,
        view: TrayMenuView,
        menu_refs: dict[str, Any],
        *,
        history_state: HistoryMenuState | None = None,
        on_health_icon: Callable[[], None] | None = None,
    ) -> None:
        self.session = session
        self.view = view
        self.menu_refs = menu_refs
        self.history_state = history_state or HistoryMenuState(signature=object())
        self.on_health_icon = on_health_icon

    @property
    def model_items(self) -> list[tuple[Any, Any]]:
        return self.menu_refs.get(MODEL_ITEMS_REF, [])

    def update_menu(self) -> None:
        apply_tray_menu_update(
            self.session,
            self.view,
            model_items=self.model_items,
            history_state=self.history_state,
        )

    def refresh_history_menu(self, *, force: bool = False) -> None:
        refresh_history_menu(
            self.session,
            self.history_state,
            self.view,
            force=force,
        )

    def run_tray_action(
        self,
        action: Callable[[], object],
        *,
        before_update: Callable[[], None] | None = None,
    ) -> object:
        return after_tray_action(
            action,
            history_state=self.history_state,
            refresh_history=lambda: self.refresh_history_menu(force=True),
            update_menu=self.update_menu,
            before_update=before_update,
        )

    def toggle_record(self) -> object:
        return self.run_tray_action(self.session.toggle_record, before_update=self.on_health_icon)

    def copy_latest(self) -> None:
        self.session.copy_latest()
        self.update_menu()

    def copy_history_text(self, text: str) -> None:
        self.session.copy_text(text)
        self.update_menu()

    def refresh_health(self) -> None:
        self.session.refresh_health()
        if self.on_health_icon is not None:
            self.on_health_icon()
        self.update_menu()


def build_tray_action_callbacks(
    controller: TrayActionHost,
    session: TraySession,
    *,
    quit: Callable[[], Any],
    set_hotkey: Callable[[], Any] | None = None,
    copy_hotkey_setup: Callable[[], Any] | None = None,
) -> dict[str, Callable[..., Any]]:
    callbacks: dict[str, Callable[..., Any]] = {
        "toggle": lambda *_: controller.toggle_record(),
        "copy_latest": lambda *_: controller.copy_latest(),
        "start_service": lambda *_: controller.run_tray_action(session.start_service),
        "restart_service": lambda *_: controller.run_tray_action(session.restart_service),
        "model_cleanup": lambda *_: controller.run_tray_action(session.toggle_model_cleanup),
        "open_settings": lambda *_: open_path(session.open_settings(), session.runtime),
        "quit": lambda *_: quit(),
    }
    if set_hotkey is not None:
        callbacks["set_hotkey"] = lambda *_: set_hotkey()
    if copy_hotkey_setup is not None:
        callbacks["copy_hotkey_setup"] = lambda *_: copy_hotkey_setup()
    return callbacks
