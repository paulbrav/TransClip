from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .history import history_file_signature
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
from .tray_session import TraySession, latest_history_text, model_menu_label, preview_text


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
    state: dict[str, Any] = {
        "history_signature": object(),
        "history_refreshing": False,
    }

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

    def _after_action(action: Callable[[], object]) -> None:
        outcome = action()
        if getattr(outcome, "latest_transcript", None):
            state["history_signature"] = object()
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
                menu_refs["history_menu_item"] = menu_item
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
        enabled = True
        if node.action == "copy_latest":
            enabled = bool(session.latest or latest_history_text())
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
        icon = icon_holder["icon"]
        if icon is None:
            return
        session.refresh_health()
        health = session.health
        menu_refs["status_item"].text = tray_status_label(health.status, health.detail)
        menu_refs["toggle_item"].text = tray_action_label(
            "toggle",
            recording=health.recording,
            settings=session.settings,
        )
        menu_refs["latest_item"].enabled = bool(session.latest or latest_history_text())
        menu_refs["model_cleanup_item"].text = tray_action_label(
            "model_cleanup",
            recording=health.recording,
            settings=session.settings,
        )
        for item, row in menu_refs[MODEL_ITEMS_REF]:
            item.text = model_menu_label(row.model_id, row.backend, session.settings)
        refresh_history_menu()
        icon.icon = build_image(health.icon)

    def refresh_history_menu(force: bool = False) -> None:
        if state["history_refreshing"]:
            return
        signature = history_file_signature()
        if not force and signature == state["history_signature"]:
            return
        state["history_refreshing"] = True
        try:
            submenu_items: list = []
            events = session.history_events()
            if events:
                for event in events:
                    text = str(event.get("text") or "")

                    def copy_history(_icon, _item, value=text):
                        session.copy_text(value)
                        update_menu()

                    submenu_items.append(pystray.MenuItem(preview_text(text), copy_history))
            else:
                submenu_items.append(pystray.MenuItem("No recent transcripts", None, enabled=False))
            menu_refs["history_menu_item"].submenu = pystray.Menu(*submenu_items)
            state["history_signature"] = signature
        finally:
            state["history_refreshing"] = False

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
