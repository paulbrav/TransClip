from __future__ import annotations

import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from .client import InferenceClient
from .daemon_lifecycle import service_action
from .history import read_history
from .models import SUPPORTED_MODELS
from .paste import SystemClipboard
from .product import DISPLAY_NAME
from .recording_ops import toggle_recording
from .settings import Settings, load_settings, settings_path, write_default_settings, write_settings
from .tray import _latest_history_text, _model_label, _model_menu_label, _open_path, _preview


def _call_on_main_thread(callback) -> None:
    from PyObjCTools.AppHelper import callAfter

    callAfter(callback)


def run_macos_tray(settings: Settings, explicit_settings_path: Path | None = None) -> int:
    try:
        import rumps
    except ImportError:
        print(
            "macOS menu bar tray requires rumps. Install: uv pip install rumps",
            file=sys.stderr,
        )
        return 1

    class TransClipMenuBarApp(rumps.App):
        def __init__(self) -> None:
            super().__init__(DISPLAY_NAME, title="🎙", quit_button=None)
            self.settings = settings
            self.explicit_settings_path = explicit_settings_path
            self.state: dict[str, Any] = {
                "status": "starting",
                "recording": False,
                "latest": "",
                "detail": "",
            }
            self._busy = False
            self.status_item = rumps.MenuItem("Starting...", callback=None)
            self.toggle_item = rumps.MenuItem("Record", callback=self.on_toggle_record)
            self.copy_item = rumps.MenuItem("Copy latest transcript", callback=self.on_copy_latest)
            self.history_menu = rumps.MenuItem("Recent transcripts")
            self.model_menu = rumps.MenuItem("ASR model")
            self.model_items: list[tuple[rumps.MenuItem, Any]] = []
            self.menu = [
                self.status_item,
                None,
                self.toggle_item,
                self.copy_item,
                self.history_menu,
                None,
                rumps.MenuItem("Start service", callback=self.on_start_service),
                rumps.MenuItem("Restart service", callback=self.on_restart_service),
                self.model_menu,
                rumps.MenuItem("Open settings", callback=self.on_open_settings),
                None,
                rumps.MenuItem("Quit", callback=self.on_quit),
            ]
            self._build_model_menu()
            self._refresh_history_menu()
            self._update_menu()
            self._timer = rumps.Timer(self.on_refresh_health, 3)
            self._timer.start()
            self.on_refresh_health(None)

        def _build_model_menu(self) -> None:
            items = []
            for model in SUPPORTED_MODELS:
                item = rumps.MenuItem(
                    _model_menu_label(model.model_id, model.backend, self.settings),
                    callback=lambda _, model_id=model.model_id, backend=model.backend: self.on_set_asr_model(
                        model_id, backend
                    ),
                )
                self.model_items.append((item, model))
                items.append(item)
            self.model_menu.update(items)

        def _refresh_history_menu(self) -> None:
            events = read_history(limit=5)
            if events:
                items = [
                    rumps.MenuItem(
                        _preview(str(event.get("text") or "")),
                        callback=lambda _, value=str(event.get("text") or ""): SystemClipboard().write(value),
                    )
                    for event in events
                ]
            else:
                items = [rumps.MenuItem("No recent transcripts", callback=None)]
            self.history_menu.update(items)

        def _update_menu(self) -> None:
            self.status_item.title = self.state["detail"] or f"Service: {self.state['status']}"
            self.toggle_item.title = "Stop + paste" if self.state["recording"] else "Record"
            has_latest = bool(self.state["latest"] or _latest_history_text())
            self.copy_item.set_callback(self.on_copy_latest if has_latest else None)
            for item, model in self.model_items:
                item.title = _model_menu_label(model.model_id, model.backend, self.settings)

        def _apply_health(self, health: dict[str, object] | None, error: Exception | None) -> None:
            if error is not None:
                self.state["status"] = "offline"
                self.state["recording"] = False
                self.state["detail"] = f"Service: offline ({error})"
                self.title = "⚠️"
            else:
                status = str((health or {}).get("status", "unknown"))
                self.state["status"] = status
                self.state["recording"] = status == "recording"
                self.state["detail"] = f"Service: {status}"
                self.title = "⏺" if self.state["recording"] else "🎙"
            self._update_menu()

        def on_refresh_health(self, _sender) -> None:
            if self._busy:
                return

            def worker() -> None:
                try:
                    health = InferenceClient(self.settings).health()
                except Exception as exc:
                    _call_on_main_thread(lambda: self._apply_health(None, exc))
                    return
                _call_on_main_thread(lambda: self._apply_health(health, None))

            threading.Thread(target=worker, daemon=True).start()

        def _finish_toggle(self, outcome) -> None:
            self._busy = False
            if not outcome.ok:
                self.state["detail"] = f"Toggle failed: {outcome.error_message}"
                self._update_menu()
                return
            if outcome.latest_transcript:
                self.state["latest"] = outcome.latest_transcript
                self._refresh_history_menu()
            if outcome.paste_failed_message:
                self.state["detail"] = outcome.paste_failed_message
            self.on_refresh_health(None)

        def _finish_toggle_error(self, exc: Exception) -> None:
            self._busy = False
            self.state["detail"] = f"Toggle failed: {exc}"
            self._update_menu()

        def on_toggle_record(self, _sender) -> None:
            if self._busy:
                return
            self._busy = True
            self.state["detail"] = "Working... (first run may load the ASR model)"
            self._update_menu()

            def worker() -> None:
                try:
                    outcome = toggle_recording(self.settings, paste=True)
                except Exception as exc:
                    _call_on_main_thread(lambda: self._finish_toggle_error(exc))
                    return
                _call_on_main_thread(lambda: self._finish_toggle(outcome))

            threading.Thread(target=worker, daemon=True).start()

        def on_copy_latest(self, _sender) -> None:
            latest = self.state["latest"] or _latest_history_text()
            if not latest:
                self.state["detail"] = "No transcript available"
                self._update_menu()
                return
            try:
                SystemClipboard().write(latest)
                self.state["detail"] = "Copied latest transcript"
            except Exception as exc:
                self.state["detail"] = f"Copy failed: {exc}"
            self._update_menu()

        def _run_service_action(self, action: str) -> None:
            if self._busy:
                return
            self._busy = True
            self.state["detail"] = f"{action.title()}ing service…"
            self._update_menu()

            def worker() -> None:
                try:
                    result = service_action(action)
                    detail = result.detail
                except Exception as exc:
                    detail = f"{action} failed: {exc}"

                def finish() -> None:
                    self._busy = False
                    self.state["detail"] = detail
                    self.on_refresh_health(None)

                _call_on_main_thread(finish)

            threading.Thread(target=worker, daemon=True).start()

        def on_start_service(self, _sender) -> None:
            self._run_service_action("start")

        def on_restart_service(self, _sender) -> None:
            self._run_service_action("restart")

        def _finish_asr_model_change(self, model_id: str, backend: str, detail: str) -> None:
            self._busy = False
            self.state["detail"] = detail
            self._build_model_menu()
            self._update_menu()

        def on_set_asr_model(self, model_id: str, backend: str) -> None:
            if self._busy:
                return
            self._busy = True
            path = self.explicit_settings_path or settings_path()

            def worker() -> None:
                try:
                    persisted = load_settings(path) if path.exists() else self.settings
                    updated = replace(persisted, asr_model=model_id, asr_backend=backend)
                    write_settings(updated, path)
                    self.settings.asr_model = model_id
                    self.settings.asr_backend = backend
                    restart = service_action("restart")
                    detail = f"ASR model set to {_model_label(model_id, backend)}; {restart.detail}"
                except Exception as exc:
                    detail = f"ASR model update failed: {exc}"
                _call_on_main_thread(lambda: self._finish_asr_model_change(model_id, backend, detail))

            threading.Thread(target=worker, daemon=True).start()

        def on_open_settings(self, _sender) -> None:
            path = self.explicit_settings_path or settings_path()
            write_default_settings(path)
            _open_path(path)

        def on_quit(self, _sender) -> None:
            rumps.quit_application()

    TransClipMenuBarApp().run()
    return 0
