from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from transclip.desktop.hotkey import install_shortcut
from transclip.product import APP_ID, DISPLAY_NAME, IMPORT_PACKAGE
from transclip.settings import Settings, patch_settings, settings_path

from .controller import TrayController, build_tray_action_callbacks
from .materialize import materialize_tray_menu
from .menu import tray_icon_for_health, tray_menu_nodes
from .menu_update import HistoryMenuState
from .session import TraySession
from .sinks.gtk import GtkMenuSink


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
                        lambda _item, value=full_text: controller.copy_history_text(value),
                    )
                item.show_all()

        def set_health_icon(self, icon: str) -> None:
            themed = tray_icon_for_health(icon, system=session.runtime.system())
            indicator.set_icon_full(themed, DISPLAY_NAME)

    menu_view = GtkMenuView()
    controller = TrayController(
        session,
        menu_view,
        menu_refs,
        history_state=history_state,
        on_health_icon=lambda: menu_view.set_health_icon(session.health.icon),
    )

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
            controller.update_menu()
            return
        if not _valid_accelerator(Gtk, value):
            session.set_detail("Hotkey is not a valid GNOME accelerator")
            controller.update_menu()
            return
        path = explicit_settings_path or settings_path()
        try:
            install_shortcut(settings_path=explicit_settings_path, binding=value)
            session.settings = patch_settings(path, hotkey_linux=value)
            session.set_detail(f"Hotkey set to {value}")
        except Exception as exc:
            session.set_detail(f"Hotkey update failed: {exc}")
        controller.update_menu()

    action_callbacks = build_tray_action_callbacks(
        controller,
        session,
        set_hotkey=set_hotkey,
        quit=Gtk.main_quit,
    )

    def build_menu() -> None:
        menu = Gtk.Menu()
        materialize_tray_menu(
            tray_menu_nodes("Linux"),
            session,
            GtkMenuSink(
                menu,
                menu_refs,
                append_separator=_append_separator,
                append_label=_append_label,
                append_item=_append_item,
                after_action=controller.run_tray_action,
                set_model=session.set_asr_model,
            ),
            action_callbacks=action_callbacks,
            on_history_open=controller.refresh_history_menu,
            history_state=history_state,
        )
        controller.refresh_history_menu()
        controller.update_menu()
        menu.show_all()
        indicator.set_menu(menu)

    build_menu()
    GLib.timeout_add_seconds(3, lambda: controller.refresh_health() or True)
    controller.refresh_health()
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


def _reexec_with_system_python(explicit_settings_path: Path | None) -> int:
    python = "/usr/bin/python3"
    if not Path(python).exists() or Path(sys.executable).resolve() == Path(python).resolve():
        print(
            "Python tray requires PyGObject/AppIndicator. Install: "
            "sudo apt install python3-gi gir1.2-ayatanaappindicator3-0.1",
            file=sys.stderr,
        )
        return 1
    from transclip.daemon.common import repo_root

    root = repo_root()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [python, "-m", f"{IMPORT_PACKAGE}.cli"]
    if explicit_settings_path:
        command.extend(["--settings", str(explicit_settings_path)])
    command.extend(["tray", "--no-system-python-fallback"])
    return subprocess.call(command, cwd=str(root), env=env)
