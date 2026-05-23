from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .gnome_shortcut import install_shortcut
from .history import history_file_signature
from .platform_runtime import open_path
from .product import APP_ID, DISPLAY_NAME, IMPORT_PACKAGE
from .settings import Settings, patch_settings, settings_path
from .tray_menu import (
    MODEL_ITEMS_REF,
    TrayMenuNode,
    asr_model_choices,
    tray_action_label,
    tray_icon_for_health,
    tray_menu_nodes,
    tray_status_label,
    tray_submenu_is_history,
    tray_submenu_title,
)
from .tray_menu_update import HistoryMenuState, apply_tray_menu_update
from .tray_menu_update import refresh_history_menu as refresh_shared_history
from .tray_session import TraySession


def run_python_tray(settings: Settings, explicit_settings_path: Path | None = None) -> int:
    try:
        import gi
    except ImportError:
        return _reexec_with_system_python(explicit_settings_path)

    gi.require_version("Gtk", "3.0")
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3 as AppIndicator
    from gi.repository import GLib, Gtk

    session = TraySession(settings, explicit_settings_path)
    history_state = HistoryMenuState(signature=object())
    menu_refs: dict[str, Any] = {}

    indicator = AppIndicator.Indicator.new(
        APP_ID,
        "audio-input-microphone-symbolic",
        AppIndicator.IndicatorCategory.APPLICATION_STATUS,
    )
    indicator.set_title(DISPLAY_NAME)
    indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)

    class GtkMenuView:
        def __init__(self) -> None:
            self._health_icon = session.health.icon

        def set_label(self, ref: str, text: str) -> None:
            _set_menu_item_label(menu_refs[ref], text)

        def set_enabled(self, ref: str, enabled: bool) -> None:
            menu_refs[ref].set_sensitive(enabled)

        def set_model_labels(self, rows) -> None:
            for item, label in rows:
                _set_menu_item_label(item, label)

        def rebuild_history(self, entries) -> None:
            history_menu = menu_refs["history_menu"]
            for child in history_menu.get_children():
                history_menu.remove(child)
            for preview, full_text in entries:
                if not full_text:
                    item = _append_label(history_menu, preview)
                else:
                    item = _append_item(
                        history_menu,
                        preview,
                        lambda _item, value=full_text: session.copy_text(value),
                    )
                item.show_all()

        def set_health_icon(self, icon: str) -> None:
            self._health_icon = icon
            themed = tray_icon_for_health(icon, system=session.runtime.system())
            indicator.set_icon_full(themed, DISPLAY_NAME)

    menu_view = GtkMenuView()

    def apply_health_icon() -> None:
        menu_view.set_health_icon(session.health.icon)

    def refresh_health() -> bool:
        session.refresh_health()
        apply_health_icon()
        update_menu()
        return True

    def toggle_record(_item=None) -> None:
        outcome = session.toggle_record()
        if outcome.latest_transcript:
            history_state.signature = object()
            refresh_history_menu(force=True)
        apply_health_icon()
        update_menu()

    def copy_latest(_item=None) -> None:
        session.copy_latest()
        update_menu()

    def _after_action(action) -> None:
        action()
        update_menu()

    def set_hotkey(_item=None) -> None:
        current = session.settings.hotkey_linux
        dialog = Gtk.Dialog(title="Set hotkey")
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Save", Gtk.ResponseType.OK)
        box = dialog.get_content_area()
        label = Gtk.Label(label='Enter a GNOME accelerator, e.g. "<Control><Alt>space"')
        entry = Gtk.Entry()
        entry.set_text(current)
        entry.set_activates_default(True)
        dialog.set_default_response(Gtk.ResponseType.OK)
        box.add(label)
        box.add(entry)
        dialog.show_all()
        response = dialog.run()
        value = entry.get_text().strip()
        dialog.destroy()
        if response != Gtk.ResponseType.OK:
            return
        if not value:
            session.set_detail("Hotkey was not changed")
            update_menu()
            return
        if not _valid_accelerator(Gtk, value):
            session.set_detail("Hotkey is not a valid GNOME accelerator")
            update_menu()
            return
        path = explicit_settings_path or settings_path()
        try:
            install_shortcut(settings_path=explicit_settings_path, binding=value)
            session.settings = patch_settings(path, hotkey_linux=value)
            session.set_detail(f"Hotkey set to {value}")
        except Exception as exc:
            session.set_detail(f"Hotkey update failed: {exc}")
        update_menu()

    action_callbacks = {
        "toggle": toggle_record,
        "copy_latest": copy_latest,
        "start_service": lambda *_: _after_action(session.start_service),
        "restart_service": lambda *_: _after_action(session.restart_service),
        "model_cleanup": lambda *_: _after_action(session.toggle_model_cleanup),
        "set_hotkey": set_hotkey,
        "open_settings": lambda *_: open_path(session.open_settings(), session.runtime),
        "quit": Gtk.main_quit,
    }

    def append_menu_node(node: TrayMenuNode, menu) -> None:
        if node.kind == "separator":
            _append_separator(menu)
            return
        if node.kind == "label":
            health = session.health
            menu_refs[node.ref] = _append_label(
                menu,
                tray_status_label(health.status, health.detail),
            )
            return
        if node.kind == "submenu":
            assert node.action is not None
            submenu = Gtk.Menu()
            menu_item = Gtk.MenuItem(label=tray_submenu_title(node.action))
            menu_item.set_submenu(submenu)
            menu.append(menu_item)
            menu_refs[node.ref] = submenu
            if tray_submenu_is_history(node.action):
                submenu.connect("map", lambda *_args: refresh_history_menu())
                return
            menu_refs[MODEL_ITEMS_REF] = []
            for label, row in asr_model_choices(session.settings, session.runtime):
                item = _append_item(
                    submenu,
                    label,
                    lambda _item, model_id=row.model_id, backend=row.backend: _after_action(
                        lambda: session.set_asr_model(model_id, backend)
                    ),
                )
                menu_refs[MODEL_ITEMS_REF].append((item, row))
            return
        assert node.action is not None
        label = tray_action_label(
            node.action,
            recording=session.health.recording,
            settings=session.settings,
        )
        item = _append_item(menu, label, action_callbacks[node.action])
        if node.ref:
            menu_refs[node.ref] = item

    def build_menu() -> None:
        menu = Gtk.Menu()
        for node in tray_menu_nodes("Linux"):
            append_menu_node(node, menu)
        refresh_history_menu()
        update_menu()
        menu.show_all()
        indicator.set_menu(menu)

    def update_menu() -> None:
        apply_tray_menu_update(
            session,
            menu_view,
            model_items=menu_refs.get(MODEL_ITEMS_REF, []),
        )

    def refresh_history_menu(force: bool = False) -> None:
        refresh_shared_history(session, history_state, menu_view, force=force)

    build_menu()
    GLib.timeout_add_seconds(3, refresh_health)
    refresh_health()
    Gtk.main()
    return 0


def _append_item(menu, label: str, callback):
    from gi.repository import Gtk

    item = Gtk.MenuItem(label=label)
    item.connect("activate", callback)
    menu.append(item)
    return item


def _append_label(menu, label: str):
    from gi.repository import Gtk

    item = Gtk.MenuItem(label=label)
    item.set_sensitive(False)
    menu.append(item)
    return item


def _set_menu_item_label(item, label: str) -> None:
    if hasattr(item, "label"):
        item.label = label
    child = item.get_child()
    if hasattr(child, "set_text"):
        child.set_text(label)
    else:
        item.set_label(label)


def _valid_accelerator(Gtk, value: str) -> bool:
    parser = getattr(Gtk, "accelerator_parse", None)
    if parser is None:
        return True
    key, _modifiers = parser(value)
    return bool(key)


def _append_separator(menu) -> None:
    from gi.repository import Gtk

    menu.append(Gtk.SeparatorMenuItem())


def _history_file_signature(path: Path | None = None) -> int | None:
    return history_file_signature(path)


def _reexec_with_system_python(explicit_settings_path: Path | None) -> int:
    python = "/usr/bin/python3"
    if not Path(python).exists() or Path(sys.executable).resolve() == Path(python).resolve():
        print(
            "Python tray requires PyGObject/AppIndicator. Install: "
            "sudo apt install python3-gi gir1.2-ayatanaappindicator3-0.1",
            file=sys.stderr,
        )
        return 1
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [python, "-m", f"{IMPORT_PACKAGE}.cli"]
    if explicit_settings_path:
        command.extend(["--settings", str(explicit_settings_path)])
    command.extend(["tray", "--no-system-python-fallback"])
    return subprocess.call(command, cwd=str(repo_root), env=env)
