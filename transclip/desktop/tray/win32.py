from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transclip.desktop.hotkey.windows import start_windows_hotkey
from transclip.platform.runtime import PlatformRuntime
from transclip.product import DISPLAY_NAME
from transclip.settings import Settings, patch_settings, settings_path

from .controller import TrayController, build_tray_action_callbacks
from .materialize import materialize_tray_menu
from .menu import tray_menu_nodes
from .menu_update import HistoryMenuState
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
        def set_label(self, ref: str, text: str) -> None:
            menu_refs[ref].text = text

        def set_enabled(self, ref: str, enabled: bool) -> None:
            menu_refs[ref].enabled = enabled

        def set_model_labels(self, rows) -> None:
            for item, label in rows:
                item.text = label

        def rebuild_history(self, entries) -> None:
            menu_refs["_history_entries"] = list(entries)

        def set_health_icon(self, icon: str) -> None:
            icon_obj = icon_holder["icon"]
            if icon_obj is not None:
                icon_obj.icon = build_image(icon)

    menu_view = PystrayMenuView()
    controller = TrayController(
        session,
        menu_view,
        menu_refs,
        history_state=history_state,
        on_health_icon=lambda: menu_view.set_health_icon(session.health.icon),
    )

    def set_hotkey(_icon=None, _item=None) -> None:
        _set_hotkey_dialog(session, restart_hotkey)
        controller.update_menu()

    action_callbacks = build_tray_action_callbacks(
        controller,
        session,
        set_hotkey=set_hotkey,
        quit=lambda: icon_holder["icon"].stop() if icon_holder["icon"] is not None else None,
    )

    def build_menu() -> pystray.Menu:
        items: list = []
        materialize_tray_menu(
            tray_menu_nodes(session.runtime.system()),
            session,
            PystrayMenuSink(
                items,
                menu_refs,
                pystray=pystray,
                after_action=controller.run_tray_action,
                set_model=session.set_asr_model,
                on_copy_history=controller.copy_history_text,
            ),
            action_callbacks=action_callbacks,
            initial_status_label=True,
            on_history_open=controller.refresh_history_menu,
            history_state=history_state,
        )
        return pystray.Menu(*items)

    def on_hotkey() -> None:
        controller.toggle_record()

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
        controller.refresh_health()
        controller.refresh_history_menu(force=True)

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
