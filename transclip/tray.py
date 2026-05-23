from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .client import InferenceClient
from .daemon_lifecycle import service_action
from .gnome_shortcut import install_shortcut, macos_hotkey_setup_message
from .history import history_path, read_history
from .models import model_rows
from .paste import SystemClipboard
from .platform_runtime import PlatformRuntime, get_runtime
from .product import APP_ID, DISPLAY_NAME, IMPORT_PACKAGE
from .recording_ops import toggle_recording
from .settings import Settings, patch_settings, settings_path, write_default_settings

_MACOS_TRAY_REFS: list[Any] = []


def run_tray(settings: Settings, explicit_settings_path: Path | None = None) -> int:
    runtime = get_runtime()
    if runtime.system() == "Darwin":
        return run_macos_tray(settings, explicit_settings_path=explicit_settings_path, runtime=runtime)
    return run_python_tray(settings, explicit_settings_path=explicit_settings_path)


def run_macos_tray(
    settings: Settings,
    explicit_settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> int:
    try:
        from AppKit import (  # type: ignore[import-not-found]
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSMenu,
            NSMenuItem,
            NSStatusBar,
            NSVariableStatusItemLength,
        )
        from Foundation import NSObject, NSTimer  # type: ignore[import-not-found]
    except ImportError:
        print(
            "macOS menu bar UI requires PyObjC. Install with: "
            "uv sync --extra macos-ui or pip install 'transclip[macos-ui]'",
            file=sys.stderr,
        )
        return 1

    platform_runtime = runtime or get_runtime()

    class MacOSTrayController(NSObject):
        def initWithSettings_explicitSettingsPath_(self, initial_settings, settings_path_value):
            self = self.init()
            self.settings = initial_settings
            self.explicit_settings_path = settings_path_value
            self.state = {
                "status": "starting",
                "recording": False,
                "latest": "",
                "detail": "",
            }
            self.menu_refs = {}
            self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            self.status_item.button().setTitle_("🎙")
            self.status_item.button().setToolTip_(DISPLAY_NAME)
            self.buildMenu()
            return self

        def buildMenu(self):
            menu = NSMenu.alloc().init()
            menu.setDelegate_(self)
            self.menu_refs["status_item"] = self.appendLabel_toMenu_(self.state["detail"] or "Service: starting", menu)
            menu.addItem_(NSMenuItem.separatorItem())
            self.menu_refs["toggle_item"] = self.appendItem_action_toMenu_("Record", "toggleRecord:", menu)
            self.menu_refs["latest_item"] = self.appendItem_action_toMenu_(
                "Copy latest transcript",
                "copyLatest:",
                menu,
            )
            menu.addItem_(NSMenuItem.separatorItem())
            history_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Recent transcripts", None, "")
            history_menu = NSMenu.alloc().init()
            history_item.setSubmenu_(history_menu)
            menu.addItem_(history_item)
            self.menu_refs["history_menu"] = history_menu
            menu.addItem_(NSMenuItem.separatorItem())
            self.appendItem_action_toMenu_("Start service", "startService:", menu)
            self.appendItem_action_toMenu_("Restart service", "restartService:", menu)
            model_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("ASR model", None, "")
            model_menu = NSMenu.alloc().init()
            model_item.setSubmenu_(model_menu)
            menu.addItem_(model_item)
            self.menu_refs["model_menu"] = model_menu
            self.menu_refs["model_items"] = []
            for row in _asr_model_rows(self.settings, platform_runtime):
                model = {"model_id": row["model_id"], "backend": row["backend"]}
                item = self.appendItem_action_toMenu_(
                    _model_menu_label(model["model_id"], model["backend"], self.settings),
                    "setASRModel:",
                    model_menu,
                )
                item.setRepresentedObject_(model)
                self.menu_refs["model_items"].append(item)
            self.appendItem_action_toMenu_("Copy hotkey setup command", "copyHotkeySetup:", menu)
            self.appendItem_action_toMenu_("Open settings", "openSettings:", menu)
            menu.addItem_(NSMenuItem.separatorItem())
            self.appendItem_action_toMenu_("Quit tray", "quitTray:", menu)
            self.refreshHistoryMenu()
            self.updateMenu()
            self.status_item.setMenu_(menu)

        def appendItem_action_toMenu_(self, title, action, menu):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
            item.setTarget_(self)
            menu.addItem_(item)
            return item

        def appendLabel_toMenu_(self, title, menu):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
            item.setEnabled_(False)
            menu.addItem_(item)
            return item

        def refreshHealth(self):
            try:
                health = InferenceClient(self.settings).health()
                status = str(health.get("status", "unknown"))
                self.state["status"] = status
                self.state["recording"] = status == "recording"
                self.state["detail"] = f"Service: {status}"
                self.status_item.button().setTitle_("●" if self.state["recording"] else "🎙")
            except Exception as exc:
                self.state["status"] = "offline"
                self.state["recording"] = False
                self.state["detail"] = f"Service: offline ({exc})"
                self.status_item.button().setTitle_("⚠")
            self.updateMenu()
            return True

        def refreshHealth_(self, _timer):
            return self.refreshHealth()

        def updateMenu(self):
            self.menu_refs["status_item"].setTitle_(self.state["detail"] or f"Service: {self.state['status']}")
            self.menu_refs["toggle_item"].setTitle_("Stop + paste" if self.state["recording"] else "Record")
            self.menu_refs["latest_item"].setEnabled_(bool(self.state["latest"] or _latest_history_text()))
            for item in self.menu_refs["model_items"]:
                model = item.representedObject()
                item.setTitle_(_model_menu_label(model["model_id"], model["backend"], self.settings))

        def refreshHistoryMenu(self):
            history_menu = self.menu_refs["history_menu"]
            for item in list(history_menu.itemArray()):
                history_menu.removeItem_(item)
            events = read_history(limit=5)
            if events:
                for event in events:
                    text = str(event.get("text") or "")
                    item = self.appendItem_action_toMenu_(_preview(text), "copyHistoryItem:", history_menu)
                    item.setRepresentedObject_(text)
            else:
                self.appendLabel_toMenu_("No recent transcripts", history_menu)

        def menuWillOpen_(self, _menu):
            self.refreshHistoryMenu()
            self.refreshHealth()

        def toggleRecord_(self, _item):
            outcome = toggle_recording(self.settings, paste=True)
            if not outcome.ok:
                self.state["detail"] = f"Toggle failed: {outcome.error_message}"
                self.updateMenu()
                return
            if outcome.latest_transcript:
                self.state["latest"] = outcome.latest_transcript
                self.refreshHistoryMenu()
            action = outcome.payload.get("action")
            detail = outcome.paste_failed_message
            self.refreshHealth()
            if action == "started":
                self.state["status"] = "recording"
                self.state["recording"] = True
                self.state["detail"] = "Service: recording"
                self.status_item.button().setTitle_("●")
            elif action == "stopped":
                self.state["recording"] = False
                if not detail:
                    self.state["detail"] = "Service: ready"
                self.status_item.button().setTitle_("🎙")
            if detail:
                self.state["detail"] = detail
            self.updateMenu()

        def copyLatest_(self, _item):
            latest = self.state["latest"] or _latest_history_text()
            if not latest:
                self.state["detail"] = "No transcript available"
                self.updateMenu()
                return
            self.copyText_(latest)

        def copyHistoryItem_(self, item):
            self.copyText_(str(item.representedObject() or ""))

        def copyText_(self, text):
            try:
                SystemClipboard().write(text)
                self.state["detail"] = "Copied latest transcript"
            except Exception as exc:
                self.state["detail"] = f"Copy failed: {exc}"
            self.updateMenu()

        def copyHotkeySetup_(self, _item):
            self.copyText_(macos_hotkey_setup_message(self.settings, self.explicit_settings_path, platform_runtime))

        def startService_(self, _item):
            result = service_action("start")
            self.refreshHealth()
            self.state["detail"] = result.detail
            self.updateMenu()

        def restartService_(self, _item):
            result = service_action("restart")
            self.refreshHealth()
            self.state["detail"] = result.detail
            self.updateMenu()

        def setASRModel_(self, item):
            model = item.representedObject()
            path = self.explicit_settings_path or settings_path()
            try:
                self.settings = patch_settings(
                    path,
                    asr_model=model["model_id"],
                    asr_backend=model["backend"],
                )
                restart = service_action("restart")
                label = _model_label(model["model_id"], model["backend"])
                self.state["detail"] = f"ASR model set to {label}; {restart.detail}"
            except Exception as exc:
                self.state["detail"] = f"ASR model update failed: {exc}"
            self.updateMenu()

        def openSettings_(self, _item):
            path = self.explicit_settings_path or settings_path()
            write_default_settings(path)
            _open_path(path)

        def quitTray_(self, _item):
            NSApplication.sharedApplication().terminate_(self)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    controller = MacOSTrayController.alloc().initWithSettings_explicitSettingsPath_(settings, explicit_settings_path)
    app.setDelegate_(controller)
    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        3.0,
        controller,
        "refreshHealth:",
        None,
        True,
    )
    _MACOS_TRAY_REFS[:] = [controller, controller.status_item, timer]
    controller.refreshHealth()
    app.run()
    return 0


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
        "history_signature": object(),
        "history_refreshing": False,
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
            state["history_signature"] = object()
            refresh_history_menu(force=True)
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
        nonlocal settings
        path = explicit_settings_path or settings_path()
        try:
            settings = patch_settings(
                path,
                asr_model=model_id,
                asr_backend=backend,
            )
            restart = service_action("restart")
            state["detail"] = f"ASR model set to {_model_label(model_id, backend)}; {restart.detail}"
        except Exception as exc:
            state["detail"] = f"ASR model update failed: {exc}"
        update_menu()

    def toggle_model_cleanup(_item=None) -> None:
        nonlocal settings
        path = explicit_settings_path or settings_path()
        try:
            next_value = not settings.voice_model_cleanup_always_on
            settings = patch_settings(
                path,
                voice_model_cleanup_always_on=next_value,
            )
            restart = service_action("restart")
            state["detail"] = f"Model cleanup always {'on' if next_value else 'off'}; {restart.detail}"
        except Exception as exc:
            state["detail"] = f"Model cleanup toggle failed: {exc}"
        update_menu()

    def open_settings(_item=None) -> None:
        path = explicit_settings_path or settings_path()
        write_default_settings(path)
        _open_path(path)

    def set_hotkey(_item=None) -> None:
        nonlocal settings
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
            install_shortcut(settings_path=explicit_settings_path, binding=value)
            settings = patch_settings(path, hotkey_linux=value)
            state["detail"] = f"Hotkey set to {value}"
        except Exception as exc:
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
        history_menu.connect("map", lambda *_args: refresh_history_menu())
        menu.append(history_item)
        _append_separator(menu)
        _append_item(menu, "Start service", start_service)
        _append_item(menu, "Restart service", restart_service)
        menu_refs["model_cleanup_item"] = _append_item(
            menu,
            _model_cleanup_label(settings),
            toggle_model_cleanup,
        )
        model_menu = Gtk.Menu()
        model_item = Gtk.MenuItem(label="ASR model")
        model_item.set_submenu(model_menu)
        menu.append(model_item)
        menu_refs["model_menu"] = model_menu
        menu_refs["model_items"] = []
        for row in _asr_model_rows(settings):
            item = _append_item(
                model_menu,
                _model_menu_label(row["model_id"], row["backend"], settings),
                lambda _item, model_id=row["model_id"], backend=row["backend"]: set_asr_model(model_id, backend),
            )
            menu_refs["model_items"].append((item, row))
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
        _set_menu_item_label(menu_refs["model_cleanup_item"], _model_cleanup_label(settings))
        for item, model in menu_refs["model_items"]:
            _set_menu_item_label(item, _model_menu_label(model["model_id"], model["backend"], settings))

    def refresh_history_menu(force: bool = False) -> None:
        if state["history_refreshing"]:
            return
        signature = _history_file_signature()
        if not force and signature == state["history_signature"]:
            return
        state["history_refreshing"] = True
        try:
            history_menu = menu_refs["history_menu"]
            for child in history_menu.get_children():
                history_menu.remove(child)
            events = read_history(limit=5)
            if events:
                for event in events:
                    text = str(event.get("text") or "")
                    item = _append_item(
                        history_menu,
                        _preview(text),
                        lambda _item, value=text: SystemClipboard().write(value),
                    )
                    item.show_all()
            else:
                item = _append_label(history_menu, "No recent transcripts")
                item.show_all()
            state["history_signature"] = signature
        finally:
            state["history_refreshing"] = False

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


def _asr_model_rows(settings: Settings, runtime: PlatformRuntime | None = None) -> list[dict[str, Any]]:
    return [row for row in model_rows(settings, runtime) if row["backend"] != "text_generation"]


def _model_menu_label(model_id: str, backend: str, settings: Settings) -> str:
    prefix = "✓ " if settings.asr_model == model_id and settings.asr_backend == backend else ""
    return prefix + _model_label(model_id, backend)


def _model_cleanup_label(settings: Settings) -> str:
    prefix = "✓ " if settings.voice_model_cleanup_always_on else ""
    return prefix + "Model cleanup always on"


def _model_label(model_id: str, backend: str) -> str:
    if backend == "granite_nar":
        return "Fast local ASR - Granite 4.1 NAR"
    if backend == "granite":
        if model_id.endswith("-plus"):
            return "Speaker/timestamp ASR - Granite 4.1 Plus"
        return "Keyword-biased ASR - Granite 4.1"
    if backend == "mlx_audio_whisper":
        return "MLX Whisper Turbo"
    if backend == "granite_mlx":
        return "MLX Granite Speech"
    return model_id


def _history_file_signature(path: Path | None = None) -> int | None:
    path = path or history_path()
    if not path.exists():
        return None
    return path.stat().st_mtime_ns


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
