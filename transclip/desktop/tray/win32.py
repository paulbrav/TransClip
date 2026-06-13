from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transclip.desktop.hotkey.windows import is_valid_hotkey, start_windows_hotkey
from transclip.desktop.win32_app import set_dpi_awareness
from transclip.platform.runtime import PlatformRuntime
from transclip.product import DISPLAY_NAME
from transclip.settings import Settings, patch_settings, settings_path

from .controller import TrayController, build_tray_action_callbacks
from .materialize import materialize_tray_menu
from .menu import tray_menu_nodes
from .menu_update import HistoryMenuState
from .session import TraySession
from .sinks.win32 import PystrayMenuSink
from .views import RefDrivenMenuView


def health_icon_color(name: str) -> str:
    """Tray icon fill for a health state, with a safe fallback.

    build_image runs inside the pystray event loop, so a KeyError here (e.g. a
    health state added later that this map does not know) would crash the tray
    rather than just show the wrong color.
    """
    return {"recording": "red", "ready": "green", "offline": "orange"}.get(name, "gray")


def run_windows_tray(
    settings: Settings,
    explicit_settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> int:
    # Opt into per-monitor-v2 DPI awareness before any window is created so the
    # tray dialog is not bitmap-stretched (blurry) on high-DPI displays.
    set_dpi_awareness()
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
        image = Image.new("RGB", (64, 64), health_icon_color(icon_name))
        draw = ImageDraw.Draw(image)
        draw.ellipse((16, 16, 48, 48), fill="white")
        return image

    def restart_hotkey() -> None:
        stop = hotkey_holder["stop"]
        if stop is not None:
            stop()
        hotkey_holder["stop"] = start_windows_hotkey(on_hotkey, session.settings, session.runtime)

    def rebuild_history(entries) -> None:
        menu_refs["_history_entries"] = list(entries)

    def set_health_icon(icon: str) -> None:
        icon_obj = icon_holder["icon"]
        if icon_obj is not None:
            icon_obj.icon = build_image(icon)

    menu_view = RefDrivenMenuView(
        menu_refs,
        set_item_label=lambda item, text: setattr(item, "text", text),
        set_item_enabled=lambda item, enabled: setattr(item, "enabled", enabled),
        rebuild_history=rebuild_history,
        set_health_icon=set_health_icon,
    )
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
    _apply_hotkey_selection(session, value, restart_hotkey)


def _apply_hotkey_selection(
    session: TraySession,
    candidate: str | None,
    restart_hotkey: Callable[[], None],
) -> None:
    if not candidate or not candidate.strip():
        session.set_detail("Hotkey was not changed")
        return
    binding = candidate.strip()
    if not is_valid_hotkey(binding):
        # Reject before persisting: an unparseable binding would otherwise be
        # saved and then crash keyboard.add_hotkey on the next registration.
        session.set_detail(f"Invalid hotkey {binding!r}; keeping {session.settings.hotkey_windows!r}")
        return
    path = session.explicit_settings_path or settings_path()
    session.settings = patch_settings(path, hotkey_windows=binding)
    restart_hotkey()
    session.set_detail(f"Hotkey set to {binding}")
