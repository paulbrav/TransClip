from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from transclip.desktop.hotkey import macos_hotkey_setup_message
from transclip.models import ModelRow
from transclip.platform.runtime import PlatformRuntime
from transclip.product import DISPLAY_NAME
from transclip.settings import Settings

from .controller import TrayController, build_tray_action_callbacks
from .materialize import materialize_tray_menu
from .menu import tray_icon_for_health, tray_menu_nodes
from .menu_update import HistoryMenuState
from .session import TraySession
from .sinks.macos import MacOSMenuSink
from .views import RefDrivenMenuView

_MACOS_TRAY_REFS: list[Any] = []


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

    session = TraySession(settings, explicit_settings_path, runtime)
    history_state = HistoryMenuState(signature=object())

    class MacOSTrayDelegate(NSObject):
        def initWithSession_(self, tray_session):
            self = self.init()
            self.session = tray_session
            self.menu_refs: dict[str, Any] = {}
            self.action_callbacks: dict[str, Any] = {}

            def rebuild_history(entries) -> None:
                history_menu = self.menu_refs["history_menu"]
                for item in list(history_menu.itemArray()):
                    history_menu.removeItem_(item)
                for preview, full_text in entries:
                    if not full_text:
                        self.appendLabel_toMenu_(preview, history_menu)
                        continue
                    item = self.appendItem_action_toMenu_(
                        preview,
                        "copyHistoryItem:",
                        history_menu,
                    )
                    item.setRepresentedObject_(full_text)

            def set_health_icon(icon: str) -> None:
                themed = tray_icon_for_health(icon, system=session.runtime.system())
                self.status_item.button().setTitle_(themed)

            self.menu_view = RefDrivenMenuView(
                self.menu_refs,
                set_item_label=lambda item, text: item.setTitle_(text),
                set_item_enabled=lambda item, enabled: item.setEnabled_(enabled),
                rebuild_history=rebuild_history,
                set_health_icon=set_health_icon,
            )
            self.tray_controller = TrayController(
                self.session,
                self.menu_view,
                self.menu_refs,
                history_state=history_state,
                on_health_icon=lambda: self.menu_view.set_health_icon(self.session.health.icon),
            )
            self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            self.status_item.button().setTitle_("🎙")
            self.status_item.button().setToolTip_(DISPLAY_NAME)
            self.buildMenu()
            return self

        def buildMenu(self):
            menu = NSMenu.alloc().init()
            menu.setDelegate_(self)
            self.action_callbacks = build_tray_action_callbacks(
                self.tray_controller,
                self.session,
                copy_hotkey_setup=lambda: self.tray_controller.copy_history_text(
                    macos_hotkey_setup_message(
                        self.session.settings,
                        self.session.explicit_settings_path,
                        self.session.runtime,
                    ),
                ),
                quit=lambda: NSApplication.sharedApplication().terminate_(self),
            )
            materialize_tray_menu(
                tray_menu_nodes(self.session.runtime.system()),
                self.session,
                MacOSMenuSink(self, menu),
                action_callbacks=self.action_callbacks,
                initial_status_label=True,
                on_history_open=self.tray_controller.refresh_history_menu,
                history_state=history_state,
            )
            self.tray_controller.refresh_history_menu(force=True)
            self.tray_controller.update_menu()
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

        def dispatchTrayAction_(self, sender):
            callback = self.action_callbacks.get(str(sender.representedObject() or ""))
            if callback is not None:
                callback()

        def refreshHealth(self):
            self.tray_controller.refresh_health()
            return True

        def refreshHealth_(self, _timer):
            return self.refreshHealth()

        def menuWillOpen_(self, _menu):
            self.tray_controller.refresh_history_menu()
            self.refreshHealth()

        def copyHistoryItem_(self, item):
            self.tray_controller.copy_history_text(str(item.representedObject() or ""))

        def setASRModel_(self, item):
            row: ModelRow = item.representedObject()
            self.tray_controller.run_tray_action(
                lambda: self.session.set_asr_model(row.model_id, row.backend),
            )

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = MacOSTrayDelegate.alloc().initWithSession_(session)
    app.setDelegate_(delegate)
    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        3.0,
        delegate,
        "refreshHealth:",
        None,
        True,
    )
    _MACOS_TRAY_REFS[:] = [delegate, delegate.status_item, timer]
    delegate.refreshHealth()
    app.run()
    return 0
