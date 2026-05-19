from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .client import InferenceClient
from .daemon_lifecycle import service_action
from .gnome_shortcut import install_shortcut
from .history import read_history
from .models import model_rows
from .paste import SystemClipboard
from .platform_runtime import get_runtime
from .product import APP_ID, CLI_COMMAND, DISPLAY_NAME, IMPORT_PACKAGE
from .recording_ops import toggle_recording
from .settings import Settings, load_settings, settings_path, write_default_settings, write_settings


def run_tray(settings: Settings, explicit_settings_path: Path | None = None) -> int:
    if get_runtime().system() == "Darwin":
        print(
            "Native macOS menu bar UI is not implemented yet. "
            f"Use a Keyboard Shortcut or Shortcuts.app action for `{CLI_COMMAND} toggle-record --paste`, "
            f"and `{CLI_COMMAND} status` / `{CLI_COMMAND} doctor` for service state.",
            file=sys.stderr,
        )
        return 1
    return run_python_tray(settings, explicit_settings_path=explicit_settings_path)


def run_python_tray(settings: Settings, explicit_settings_path: Path | None = None) -> int:
    try:
        import gi
    except ImportError:
        return _reexec_with_system_python(explicit_settings_path)

    gi.require_version("Gtk", "3.0")
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3 as AppIndicator
    from gi.repository import GLib, Gtk

    state: dict[str, Any] = {
        "status": "starting",
        "recording": False,
        "latest": "",
        "detail": "",
    }

    indicator = AppIndicator.Indicator.new(
        APP_ID,
        "audio-input-microphone-symbolic",
        AppIndicator.IndicatorCategory.APPLICATION_STATUS,
    )
    indicator.set_title(DISPLAY_NAME)
    indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
    menu_refs: dict[str, Any] = {}

    def refresh_health() -> bool:
        try:
            health = InferenceClient(settings).health()
            status = str(health.get("status", "unknown"))
            state["status"] = status
            state["recording"] = status == "recording"
            state["detail"] = f"Service: {status}"
            indicator.set_icon_full(
                "media-record-symbolic" if state["recording"] else "audio-input-microphone-symbolic",
                DISPLAY_NAME,
            )
        except Exception as exc:
            state["status"] = "offline"
            state["recording"] = False
            state["detail"] = f"Service: offline ({exc})"
            indicator.set_icon_full("dialog-warning-symbolic", f"{DISPLAY_NAME} offline")
        update_menu()
        return True

    def toggle_record(_item=None) -> None:
        outcome = toggle_recording(settings, paste=True)
        if not outcome.ok:
            state["detail"] = f"Toggle failed: {outcome.error_message}"
            update_menu()
            return
        if outcome.latest_transcript:
            state["latest"] = outcome.latest_transcript
            refresh_history_menu()
        detail = outcome.paste_failed_message
        refresh_health()
        if detail:
            state["detail"] = detail
            update_menu()

    def copy_latest(_item=None) -> None:
        latest = state["latest"] or _latest_history_text()
        if not latest:
            state["detail"] = "No transcript available"
            update_menu()
            return
        try:
            SystemClipboard().write(latest)
            state["detail"] = "Copied latest transcript"
        except Exception as exc:
            state["detail"] = f"Copy failed: {exc}"
        update_menu()

    def start_service(_item=None) -> None:
        result = service_action("start")
        state["detail"] = result.detail
        refresh_health()

    def restart_service(_item=None) -> None:
        result = service_action("restart")
        state["detail"] = result.detail
        refresh_health()

    def set_asr_model(model_id: str, backend: str) -> None:
        path = explicit_settings_path or settings_path()
        try:
            persisted = load_settings(path) if path.exists() else settings
            updated = replace(persisted, asr_model=model_id, asr_backend=backend)
            write_settings(updated, path)
            settings.asr_model = model_id
            settings.asr_backend = backend
            restart = service_action("restart")
            state["detail"] = f"ASR model set to {_model_label(model_id, backend)}; {restart.detail}"
        except Exception as exc:
            state["detail"] = f"ASR model update failed: {exc}"
        update_menu()

    def open_settings(_item=None) -> None:
        path = explicit_settings_path or settings_path()
        write_default_settings(path)
        _open_path(path)

    def set_hotkey(_item=None) -> None:
        current = settings.hotkey_linux
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
            state["detail"] = "Hotkey was not changed"
            update_menu()
            return
        if not _valid_accelerator(Gtk, value):
            state["detail"] = "Hotkey is not a valid GNOME accelerator"
            update_menu()
            return
        path = explicit_settings_path or settings_path()
        try:
            persisted = load_settings(path) if path.exists() else settings
            updated = replace(persisted, hotkey_linux=value)
            write_settings(updated, path)
            install_shortcut(settings_path=explicit_settings_path, binding=value)
            settings.hotkey_linux = value
            state["detail"] = f"Hotkey set to {value}"
        except Exception as exc:
            if "persisted" in locals():
                write_settings(persisted, path)
            state["detail"] = f"Hotkey update failed: {exc}"
        update_menu()

    def quit_tray(_item=None) -> None:
        Gtk.main_quit()

    def build_menu() -> None:
        menu = Gtk.Menu()
        menu_refs["status_item"] = _append_label(menu, state["detail"] or f"Service: {state['status']}")
        _append_separator(menu)
        menu_refs["toggle_item"] = _append_item(
            menu,
            "Stop + paste" if state["recording"] else "Record",
            toggle_record,
        )
        menu_refs["latest_item"] = _append_item(menu, "Copy latest transcript", copy_latest)
        _append_separator(menu)
        history_menu = Gtk.Menu()
        menu_refs["history_menu"] = history_menu
        history_item = Gtk.MenuItem(label="Recent transcripts")
        history_item.set_submenu(history_menu)
        history_item.connect("select", lambda _item: refresh_history_menu())
        menu.append(history_item)
        _append_separator(menu)
        _append_item(menu, "Start service", start_service)
        _append_item(menu, "Restart service", restart_service)
        model_menu = Gtk.Menu()
        model_item = Gtk.MenuItem(label="ASR model")
        model_item.set_submenu(model_menu)
        menu.append(model_item)
        menu_refs["model_menu"] = model_menu
        menu_refs["model_items"] = []
        for row in model_rows(settings):
            model = type("ModelMenuEntry", (), {"model_id": row["model_id"], "backend": row["backend"]})()
            item = _append_item(
                model_menu,
                _model_menu_label(model.model_id, model.backend, settings),
                lambda _item, model_id=model.model_id, backend=model.backend: set_asr_model(model_id, backend),
            )
            menu_refs["model_items"].append((item, model))
        _append_item(menu, "Set hotkey...", set_hotkey)
        _append_item(menu, "Open settings", open_settings)
        _append_separator(menu)
        _append_item(menu, "Quit tray", quit_tray)
        refresh_history_menu()
        update_menu()
        menu.show_all()
        indicator.set_menu(menu)

    def update_menu() -> None:
        _set_menu_item_label(menu_refs["status_item"], state["detail"] or f"Service: {state['status']}")
        _set_menu_item_label(menu_refs["toggle_item"], "Stop + paste" if state["recording"] else "Record")
        menu_refs["latest_item"].set_sensitive(bool(state["latest"] or _latest_history_text()))
        for item, model in menu_refs["model_items"]:
            _set_menu_item_label(item, _model_menu_label(model.model_id, model.backend, settings))

    def refresh_history_menu() -> None:
        history_menu = menu_refs["history_menu"]
        for child in history_menu.get_children():
            history_menu.remove(child)
        events = read_history(limit=5)
        if events:
            for event in events:
                text = str(event.get("text") or "")
                _append_item(history_menu, _preview(text), lambda _item, value=text: SystemClipboard().write(value))
        else:
            _append_label(history_menu, "No recent transcripts")
        history_menu.show_all()

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


def _model_menu_label(model_id: str, backend: str, settings: Settings) -> str:
    prefix = "✓ " if settings.asr_model == model_id and settings.asr_backend == backend else ""
    return prefix + _model_label(model_id, backend)


def _model_label(model_id: str, backend: str) -> str:
    if backend == "granite_nar":
        return "Fast local ASR - Granite 4.1 NAR"
    if backend == "granite":
        if model_id.endswith("-plus"):
            return "Speaker/timestamp ASR - Granite 4.1 Plus"
        return "Keyword-biased ASR - Granite 4.1"
    return model_id


def _latest_history_text() -> str:
    events = read_history(limit=1)
    return str(events[0].get("text") or "") if events else ""


def _preview(text: str, limit: int = 48) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "..."


def _open_path(path: Path) -> None:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
