from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transclip.desktop.hotkey.windows import start_windows_hotkey
from transclip.platform.runtime import PlatformRuntime, open_path
from transclip.product import DISPLAY_NAME
from transclip.settings import Settings, patch_settings, settings_path

from .materialize import materialize_tray_menu
from .menu import MODEL_ITEMS_REF, tray_menu_nodes
from .menu_update import (
    HistoryMenuState,
    after_tray_action,
    apply_tray_menu_update,
)
from .menu_update import refresh_history_menu as refresh_shared_history
from .session import TraySession
from .sinks.win32 import PystrayMenuSink


def run_windows_tray(
    settings: Settings,
    explicit_settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> int:
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        print(
            "Windows tray UI requires pystray and Pillow. Install with: "
            "uv sync --extra windows-ui or pip install 'transclip[windows-ui]'",
            file=sys.stderr,
        )
        return 1

    session = TraySession(settings, explicit_settings_path, runtime)
    icon_holder: dict[str, object] = {"icon": None}
    hotkey_holder: dict[str, Callable[[], None] | None] = {"stop": None}
    menu_refs: dict[str, Any] = {}
    history_state = HistoryMenuState(signature=object())

    def build_image(icon_name: str):
        color = {"recording": "red", "ready": "green", "offline": "orange"}[icon_name]
        image = Image.new("RGB", (64, 64), color)
        draw = ImageDraw.Draw(image)
        draw.ellipse((16, 16, 48, 48), fill="white")
        return image

    def restart_hotkey() -> None:
        stop = hotkey_holder["stop"]
        if stop is not None:
            stop()
        hotkey_holder["stop"] = start_windows_hotkey(on_hotkey, session.settings, session.runtime)

    class PystrayMenuView:
        def __init__(self) -> None:
            self._health_icon = session.health.icon

        def set_label(self, ref: str, text: str) -> None:
            menu_refs[ref].text = text

        def set_enabled(self, ref: str, enabled: bool) -> None:
            menu_refs[ref].enabled = enabled

        def set_model_labels(self, rows) -> None:
            for item, label in rows:
                item.text = label

        def rebuild_history(self, entries) -> None:
            submenu_items: list = []
            for preview, full_text in entries:
                if not full_text:
                    submenu_items.append(pystray.MenuItem(preview, None, enabled=False))
                    continue

                def copy_history(_icon, _item, value=full_text):
                    session.copy_text(value)
                    update_menu()

                submenu_items.append(pystray.MenuItem(preview, copy_history))
            menu_refs["history_menu"].submenu = pystray.Menu(*submenu_items)

        def set_health_icon(self, icon: str) -> None:
            self._health_icon = icon
            icon_obj = icon_holder["icon"]
            if icon_obj is not None:
                icon_obj.icon = build_image(icon)

    menu_view = PystrayMenuView()

    def run_tray_action(action: Callable[[], object]) -> None:
        after_tray_action(
            action,
            history_state=history_state,
            refresh_history=lambda: refresh_history_menu(force=True),
            update_menu=update_menu,
        )

    def toggle_record(_icon=None, _item=None) -> None:
        run_tray_action(session.toggle_record)

    def copy_latest(_icon=None, _item=None) -> None:
        session.copy_latest()
        update_menu()

    def set_hotkey(_icon=None, _item=None) -> None:
        _set_hotkey_dialog(session, restart_hotkey)
        update_menu()

    action_callbacks = {
        "toggle": toggle_record,
        "copy_latest": copy_latest,
        "start_service": lambda *_: run_tray_action(session.start_service),
        "restart_service": lambda *_: run_tray_action(session.restart_service),
        "model_cleanup": lambda *_: run_tray_action(session.toggle_model_cleanup),
        "set_hotkey": set_hotkey,
        "open_settings": lambda *_: open_path(session.open_settings(), session.runtime),
        "quit": lambda *_: icon_holder["icon"].stop() if icon_holder["icon"] is not None else None,
    }

    def build_menu() -> pystray.Menu:
        items: list = []
        materialize_tray_menu(
            tray_menu_nodes(session.runtime.system()),
            session,
            PystrayMenuSink(
                items,
                menu_refs,
                pystray=pystray,
                after_action=run_tray_action,
                set_model=session.set_asr_model,
            ),
            action_callbacks=action_callbacks,
            initial_status_label=True,
        )
        return pystray.Menu(*items)

    def update_menu() -> None:
        if icon_holder["icon"] is None:
            return
        apply_tray_menu_update(
            session,
            menu_view,
            model_items=menu_refs.get(MODEL_ITEMS_REF, []),
        )
        refresh_history_menu()

    def refresh_history_menu(force: bool = False) -> None:
        refresh_shared_history(session, history_state, menu_view, force=force)

    def on_hotkey() -> None:
        toggle_record()

    icon = pystray.Icon(
        DISPLAY_NAME,
        build_image(session.health.icon),
        DISPLAY_NAME,
        build_menu(),
    )
    icon_holder["icon"] = icon

    def setup(_icon) -> None:
        restart_hotkey()
        _icon.visible = True
        session.refresh_health()
        refresh_history_menu(force=True)
        update_menu()

    try:
        icon.run(setup=setup)
    finally:
        stop = hotkey_holder["stop"]
        if stop is not None:
            stop()
    return 0


def _set_hotkey_dialog(session: TraySession, restart_hotkey: Callable[[], None]) -> None:
    try:
        import tkinter as tk
        from tkinter import simpledialog
    except ImportError:
        session.set_detail("tkinter is unavailable for hotkey dialog")
        return
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askstring(
        "Set hotkey",
        "Enter a keyboard-library hotkey, e.g. ctrl+shift+space",
        initialvalue=session.settings.hotkey_windows,
        parent=root,
    )
    root.destroy()
    if not value:
        session.set_detail("Hotkey was not changed")
        return
    path = session.explicit_settings_path or settings_path()
    binding = value.strip()
    session.settings = patch_settings(path, hotkey_windows=binding)
    restart_hotkey()
    session.set_detail(f"Hotkey set to {binding}")
