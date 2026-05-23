from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .models import ModelRow
from .platform_runtime import PlatformRuntime, open_path
from .product import DISPLAY_NAME
from .settings import Settings
from .tray_menu import (
    MACOS_SELECTORS,
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

        def buildMenu(self):
            menu = NSMenu.alloc().init()
            menu.setDelegate_(self)
            for node in tray_menu_nodes(self.session.runtime.system()):
                self._append_menu_node(node, menu)
            self.refreshHistoryMenu(force=True)
            self.updateMenu()
            self.status_item.setMenu_(menu)

        def _append_menu_node(self, node: TrayMenuNode, menu) -> None:
            if node.kind == "separator":
                menu.addItem_(NSMenuItem.separatorItem())
                return
            if node.kind == "label":
                health = self.session.health
                self.menu_refs[node.ref] = self.appendLabel_toMenu_(
                    tray_status_label(health.status, health.detail, initial=True),
                    menu,
                )
                return
            if node.kind == "submenu":
                assert node.action is not None
                submenu = NSMenu.alloc().init()
                item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    tray_submenu_title(node.action),
                    None,
                    "",
                )
                item.setSubmenu_(submenu)
                menu.addItem_(item)
                self.menu_refs[node.ref] = submenu
                if tray_submenu_is_history(node.action):
                    return
                self.menu_refs[MODEL_ITEMS_REF] = []
                for label, row in asr_model_choices(self.session.settings, self.session.runtime):
                    model_item = self.appendItem_action_toMenu_(label, "setASRModel:", submenu)
                    model_item.setRepresentedObject_(row)
                    self.menu_refs[MODEL_ITEMS_REF].append(model_item)
                return
            assert node.action is not None
            label = tray_action_label(
                node.action,
                recording=self.session.health.recording,
                settings=self.session.settings,
            )
            item = self.appendItem_action_toMenu_(label, MACOS_SELECTORS[node.action], menu)
            if node.ref:
                self.menu_refs[node.ref] = item

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
            model_items = [
                (item, item.representedObject())
                for item in self.menu_refs.get(MODEL_ITEMS_REF, [])
            ]
            apply_tray_menu_update(self.session, self.menu_view, model_items=model_items)

        def refreshHistoryMenu(self, force: bool = False):
            refresh_shared_history(self.session, history_state, self.menu_view, force=force)

        def menuWillOpen_(self, _menu):
            self.refreshHistoryMenu()
            self.refreshHealth()

        def toggleRecord_(self, _item):
            outcome = self.session.toggle_record()
            if outcome.latest_transcript:
                history_state.signature = object()
                self.refreshHistoryMenu(force=True)
            self.updateMenu()

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
            self.session.start_service()
            self.updateMenu()

        def restartService_(self, _item):
            self.session.restart_service()
            self.updateMenu()

        def toggleModelCleanup_(self, _item):
            self.session.toggle_model_cleanup()
            self.updateMenu()

        def setASRModel_(self, item):
            row: ModelRow = item.representedObject()
            self.session.set_asr_model(row.model_id, row.backend)
            self.updateMenu()

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
