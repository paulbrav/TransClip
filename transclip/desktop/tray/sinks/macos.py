from __future__ import annotations

from ..menu import MODEL_ITEMS_REF


class MacOSMenuSink:
    def __init__(self, controller, target_menu) -> None:
        self._controller = controller
        self._menu = target_menu

    def separator(self) -> None:
        from AppKit import NSMenuItem  # type: ignore[import-not-found]

        self._menu.addItem_(NSMenuItem.separatorItem())

    def status_label(self, ref: str, text: str) -> None:
        self._controller.menu_refs[ref] = self._controller.appendLabel_toMenu_(text, self._menu)

    def action(self, ref: str, label: str, action, *, enabled: bool = True, callback=None) -> None:
        if callback is None:
            raise ValueError("macOS tray actions require callbacks")
        key = ref or str(action)
        self._controller.action_callbacks[key] = callback
        item = self._controller.appendItem_action_toMenu_(
            label,
            "dispatchTrayAction:",
            self._menu,
        )
        item.setRepresentedObject_(key)
        item.setEnabled_(enabled)
        if ref:
            self._controller.menu_refs[ref] = item

    def history_submenu(self, ref: str, title: str, on_open=None) -> None:
        del on_open
        from AppKit import NSMenu, NSMenuItem  # type: ignore[import-not-found]

        submenu = NSMenu.alloc().init()
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
        item.setSubmenu_(submenu)
        self._menu.addItem_(item)
        self._controller.menu_refs[ref] = submenu

    def model_submenu(self, ref: str, title: str, choices) -> None:
        from AppKit import NSMenu, NSMenuItem  # type: ignore[import-not-found]

        submenu = NSMenu.alloc().init()
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
        item.setSubmenu_(submenu)
        self._menu.addItem_(item)
        self._controller.menu_refs[ref] = submenu
        self._controller.menu_refs[MODEL_ITEMS_REF] = []
        for label, row in choices:
            model_item = self._controller.appendItem_action_toMenu_(label, "setASRModel:", submenu)
            model_item.setRepresentedObject_(row)
            self._controller.menu_refs[MODEL_ITEMS_REF].append((model_item, row))
