from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .hotkey_windows import start_windows_hotkey
from .platform_runtime import PlatformRuntime, open_path
from .product import DISPLAY_NAME
from .settings import Settings, patch_settings, settings_path
from .tray_menu import (
    MODEL_ITEMS_REF,
    TrayMenuNode,
    asr_model_choices,
    tray_action_label,
    tray_menu_nodes,
    tray_status_label,
    tray_submenu_is_history,
    tray_submenu_title,
)
from .tray_menu_update import HistoryMenuState, apply_tray_menu_update
from .tray_menu_update import refresh_history_menu as refresh_shared_history
from .tray_session import TraySession


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

    def _after_action(action: Callable[[], object]) -> None:
        outcome = action()
        if getattr(outcome, "latest_transcript", None):
            history_state.signature = object()
            refresh_history_menu(force=True)
        update_menu()

    def toggle_record(_icon=None, _item=None) -> None:
        _after_action(session.toggle_record)

    def copy_latest(_icon=None, _item=None) -> None:
        session.copy_latest()
        update_menu()

    def set_hotkey(_icon=None, _item=None) -> None:
        _set_hotkey_dialog(session, restart_hotkey)
        update_menu()

    action_callbacks = {
        "toggle": toggle_record,
        "copy_latest": copy_latest,
        "start_service": lambda *_: _after_action(session.start_service),
        "restart_service": lambda *_: _after_action(session.restart_service),
        "model_cleanup": lambda *_: _after_action(session.toggle_model_cleanup),
        "set_hotkey": set_hotkey,
        "open_settings": lambda *_: open_path(session.open_settings(), session.runtime),
        "quit": lambda *_: icon_holder["icon"].stop() if icon_holder["icon"] is not None else None,
    }

    def append_menu_node(node: TrayMenuNode, items: list) -> None:
        if node.kind == "separator":
            items.append(pystray.Menu.SEPARATOR)
            return
        if node.kind == "label":
            health = session.health
            item = pystray.MenuItem(
                tray_status_label(health.status, health.detail, initial=True),
                None,
                enabled=False,
            )
            items.append(item)
            menu_refs[node.ref] = item
            return
        if node.kind == "submenu":
            assert node.action is not None
            submenu_items: list = []
            if tray_submenu_is_history(node.action):
                menu_item = pystray.MenuItem(tray_submenu_title(node.action), pystray.Menu(*submenu_items))
                items.append(menu_item)
                menu_refs["history_menu"] = menu_item
                return
            menu_refs[MODEL_ITEMS_REF] = []
            for label, row in asr_model_choices(session.settings, session.runtime):

                def set_model(_icon, _item, model_id=row.model_id, backend=row.backend):
                    _after_action(lambda: session.set_asr_model(model_id, backend))

                model_item = pystray.MenuItem(label, set_model)
                submenu_items.append(model_item)
                menu_refs[MODEL_ITEMS_REF].append((model_item, row))
            items.append(pystray.MenuItem(tray_submenu_title(node.action), pystray.Menu(*submenu_items)))
            return
        assert node.action is not None
        label = tray_action_label(
            node.action,
            recording=session.health.recording,
            settings=session.settings,
        )
        enabled = node.action != "copy_latest" or bool(session.latest)
        item = pystray.MenuItem(label, action_callbacks[node.action], enabled=enabled)
        items.append(item)
        if node.ref:
            menu_refs[node.ref] = item

    def build_menu() -> pystray.Menu:
        items: list = []
        for node in tray_menu_nodes(session.runtime.system()):
            append_menu_node(node, items)
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
