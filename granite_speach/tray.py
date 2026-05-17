from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .client import InferenceClient
from .daemon_lifecycle import service_action
from .history import read_history
from .paste import SystemClipboard
from .recording_ops import toggle_recording
from .settings import Settings, keywords_path, settings_path, write_default_keywords, write_default_settings

APP_ID = "granite-speach"


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
    indicator.set_title("Granite Speach")
    indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)

    def refresh_health() -> bool:
        try:
            health = InferenceClient(settings).health()
            status = str(health.get("status", "unknown"))
            state["status"] = status
            state["recording"] = status == "recording"
            state["detail"] = f"Service: {status}"
            indicator.set_icon_full(
                "media-record-symbolic" if state["recording"] else "audio-input-microphone-symbolic",
                "Granite Speach",
            )
        except Exception as exc:
            state["status"] = "offline"
            state["recording"] = False
            state["detail"] = f"Service: offline ({exc})"
            indicator.set_icon_full("dialog-warning-symbolic", "Granite Speach offline")
        rebuild_menu()
        return True

    def toggle_record(_item=None) -> None:
        outcome = toggle_recording(settings, paste=True)
        if not outcome.ok:
            state["detail"] = f"Toggle failed: {outcome.error_message}"
            rebuild_menu()
            return
        result = outcome.payload
        if result.get("action") == "stopped" and result.get("text"):
            state["latest"] = str(result["text"])
        detail = outcome.paste_failed_message
        refresh_health()
        if detail:
            state["detail"] = detail
            rebuild_menu()

    def copy_latest(_item=None) -> None:
        latest = state["latest"] or _latest_history_text()
        if not latest:
            state["detail"] = "No transcript available"
            rebuild_menu()
            return
        try:
            SystemClipboard().write(latest)
            state["detail"] = "Copied latest transcript"
        except Exception as exc:
            state["detail"] = f"Copy failed: {exc}"
        rebuild_menu()

    def start_service(_item=None) -> None:
        result = service_action("start")
        state["detail"] = result.detail
        refresh_health()

    def restart_service(_item=None) -> None:
        result = service_action("restart")
        state["detail"] = result.detail
        refresh_health()

    def open_settings(_item=None) -> None:
        path = explicit_settings_path or settings_path()
        write_default_settings(path)
        _open_path(path)

    def open_keywords(_item=None) -> None:
        config_dir = explicit_settings_path.parent if explicit_settings_path else None
        path = keywords_path(config_dir)
        write_default_keywords(path)
        _open_path(path)

    def quit_tray(_item=None) -> None:
        Gtk.main_quit()

    def rebuild_menu() -> None:
        menu = Gtk.Menu()
        _append_label(menu, state["detail"] or f"Service: {state['status']}")
        _append_separator(menu)
        _append_item(menu, "Stop + paste" if state["recording"] else "Record", toggle_record)
        latest_item = _append_item(menu, "Copy latest transcript", copy_latest)
        latest_item.set_sensitive(bool(state["latest"] or _latest_history_text()))
        _append_separator(menu)
        history_menu = Gtk.Menu()
        events = read_history(limit=5)
        if events:
            for event in events:
                text = str(event.get("text") or "")
                _append_item(history_menu, _preview(text), lambda _item, value=text: SystemClipboard().write(value))
        else:
            _append_label(history_menu, "No recent transcripts")
        history_item = Gtk.MenuItem(label="Recent transcripts")
        history_item.set_submenu(history_menu)
        menu.append(history_item)
        _append_separator(menu)
        _append_item(menu, "Start service", start_service)
        _append_item(menu, "Restart service", restart_service)
        _append_item(menu, "Open settings", open_settings)
        _append_item(menu, "Open keyword glossary", open_keywords)
        _append_separator(menu)
        _append_item(menu, "Quit tray", quit_tray)
        menu.show_all()
        indicator.set_menu(menu)

    rebuild_menu()
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


def _append_separator(menu) -> None:
    from gi.repository import Gtk

    menu.append(Gtk.SeparatorMenuItem())


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
    command = [python, "-m", "granite_speach.cli"]
    if explicit_settings_path:
        command.extend(["--settings", str(explicit_settings_path)])
    command.extend(["tray", "--no-system-python-fallback"])
    return subprocess.call(command, cwd=str(repo_root), env=env)
