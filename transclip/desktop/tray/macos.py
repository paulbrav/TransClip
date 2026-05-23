from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from transclip.models import ModelRow
from transclip.platform.runtime import PlatformRuntime, open_path
from transclip.product import DISPLAY_NAME
from transclip.settings import Settings

from .materialize import materialize_tray_menu
from .menu import (
    MODEL_ITEMS_REF,
    tray_icon_for_health,
    tray_menu_nodes,
)
from .menu_update import (
    HistoryMenuState,
    after_tray_action,
    apply_tray_menu_update,
)
from .menu_update import refresh_history_menu as refresh_shared_history
from .session import TraySession
from .sinks.macos import MacOSMenuSink

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

    class MacOSMenuView:
        def __init__(self, controller) -> None:
            self._controller = controller

        def set_label(self, ref: str, text: str) -> None:
            self._controller.menu_refs[ref].setTitle_(text)

        def set_enabled(self, ref: str, enabled: bool) -> None:
            self._controller.menu_refs[ref].setEnabled_(enabled)

        def set_model_labels(self, rows) -> None:
            for item, label in rows:
                item.setTitle_(label)

        def rebuild_history(self, entries) -> None:
            history_menu = self._controller.menu_refs["history_menu"]
            for item in list(history_menu.itemArray()):
                history_menu.removeItem_(item)
            for preview, full_text in entries:
                if not full_text:
                    self._controller.appendLabel_toMenu_(preview, history_menu)
                    continue
                item = self._controller.appendItem_action_toMenu_(
                    preview,
                    "copyHistoryItem:",
                    history_menu,
                )
                item.setRepresentedObject_(full_text)

        def set_health_icon(self, icon: str) -> None:
            themed = tray_icon_for_health(icon, system=session.runtime.system())
            self._controller.status_item.button().setTitle_(themed)

    class MacOSTrayController(NSObject):
        def initWithSession_(self, tray_session):
            self = self.init()
            self.session = tray_session
            self.menu_refs: dict[str, Any] = {}
            self.menu_view = MacOSMenuView(self)
            self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            self.status_item.button().setTitle_("🎙")
            self.status_item.button().setToolTip_(DISPLAY_NAME)
            self.buildMenu()
            return self

        def runTrayAction_(self, action):
            after_tray_action(
                action,
                history_state=history_state,
                refresh_history=lambda: self.refreshHistoryMenu(force=True),
                update_menu=self.updateMenu,
            )

        def buildMenu(self):
            menu = NSMenu.alloc().init()
            menu.setDelegate_(self)
            materialize_tray_menu(
                tray_menu_nodes(self.session.runtime.system()),
                self.session,
                MacOSMenuSink(self, menu),
                action_callbacks={},
                initial_status_label=True,
            )
            self.refreshHistoryMenu(force=True)
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
            self.session.refresh_health()
            self.updateMenu()
            return True

        def refreshHealth_(self, _timer):
            return self.refreshHealth()

        def updateMenu(self):
            apply_tray_menu_update(
                self.session,
                self.menu_view,
                model_items=self.menu_refs.get(MODEL_ITEMS_REF, []),
            )

        def refreshHistoryMenu(self, force: bool = False):
            refresh_shared_history(self.session, history_state, self.menu_view, force=force)

        def menuWillOpen_(self, _menu):
            self.refreshHistoryMenu()
            self.refreshHealth()

        def toggleRecord_(self, _item):
            self.runTrayAction_(self.session.toggle_record)

        def copyLatest_(self, _item):
            self.session.copy_latest()
            self.updateMenu()

        def copyHistoryItem_(self, item):
            self.session.copy_text(str(item.representedObject() or ""))
            self.updateMenu()

        def copyHotkeySetup_(self, _item):
            self.session.copy_text(self.session.hotkey_setup_message())
            self.updateMenu()

        def startService_(self, _item):
            self.runTrayAction_(self.session.start_service)

        def restartService_(self, _item):
            self.runTrayAction_(self.session.restart_service)

        def toggleModelCleanup_(self, _item):
            self.runTrayAction_(self.session.toggle_model_cleanup)

        def setASRModel_(self, item):
            row: ModelRow = item.representedObject()
            self.runTrayAction_(lambda: self.session.set_asr_model(row.model_id, row.backend))

        def openSettings_(self, _item):
            open_path(self.session.open_settings(), self.session.runtime)

        def quitTray_(self, _item):
            NSApplication.sharedApplication().terminate_(self)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    controller = MacOSTrayController.alloc().initWithSession_(session)
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
