from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..menu import MODEL_ITEMS_REF


class PystrayMenuSink:
    def __init__(
        self,
        items: list,
        menu_refs: dict[str, Any],
        *,
        pystray,
        after_action: Callable[[Callable[[], object]], None],
        set_model: Callable[[str, str], None],
    ) -> None:
        self._items = items
        self._menu_refs = menu_refs
        self._pystray = pystray
        self._after_action = after_action
        self._set_model = set_model

    def separator(self) -> None:
        self._items.append(self._pystray.Menu.SEPARATOR)

    def status_label(self, ref: str, text: str) -> None:
        item = self._pystray.MenuItem(text, None, enabled=False)
        self._items.append(item)
        self._menu_refs[ref] = item

    def action(self, ref: str, label: str, action, *, enabled: bool = True, callback=None) -> None:
        item = self._pystray.MenuItem(label, callback, enabled=enabled)
        self._items.append(item)
        if ref:
            self._menu_refs[ref] = item

    def history_submenu(self, ref: str, title: str, on_open=None) -> None:
        del on_open, ref
        menu_item = self._pystray.MenuItem(title, self._pystray.Menu())
        self._items.append(menu_item)
        self._menu_refs["history_menu"] = menu_item

    def model_submenu(self, ref: str, title: str, choices) -> None:
        del ref
        submenu_items: list = []
        self._menu_refs[MODEL_ITEMS_REF] = []
        for label, row in choices:

            def set_model(_icon, _item, model_id=row.model_id, backend=row.backend):
                self._after_action(lambda: self._set_model(model_id, backend))

            model_item = self._pystray.MenuItem(label, set_model)
            submenu_items.append(model_item)
            self._menu_refs[MODEL_ITEMS_REF].append((model_item, row))
        self._items.append(self._pystray.MenuItem(title, self._pystray.Menu(*submenu_items)))
