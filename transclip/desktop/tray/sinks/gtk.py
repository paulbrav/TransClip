from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..menu import MODEL_ITEMS_REF


class GtkMenuSink:
    def __init__(
        self,
        menu,
        menu_refs: dict[str, Any],
        *,
        append_separator: Callable[[Any], None],
        append_label: Callable[[Any, str], Any],
        append_item: Callable[[Any, str, Callable[..., Any]], Any],
        after_action: Callable[[Callable[[], object]], None],
        set_model: Callable[[str, str], None],
    ) -> None:
        self._menu = menu
        self._menu_refs = menu_refs
        self._append_separator = append_separator
        self._append_label = append_label
        self._append_item = append_item
        self._after_action = after_action
        self._set_model = set_model

    def separator(self) -> None:
        self._append_separator(self._menu)

    def status_label(self, ref: str, text: str) -> None:
        item = self._append_label(self._menu, text)
        item.set_sensitive(False)
        self._menu_refs[ref] = item

    def action(self, ref: str, label: str, action, *, enabled: bool = True, callback=None) -> None:
        item = self._append_item(self._menu, label, callback)
        item.set_sensitive(enabled)
        if ref:
            self._menu_refs[ref] = item

    def history_submenu(self, ref: str, title: str, on_open=None) -> None:
        from gi.repository import Gtk

        submenu = Gtk.Menu()
        menu_item = Gtk.MenuItem(label=title)
        menu_item.set_submenu(submenu)
        self._menu.append(menu_item)
        self._menu_refs[ref] = submenu
        if on_open is not None:
            submenu.connect("map", lambda *_args: on_open())

    def model_submenu(self, ref: str, title: str, choices) -> None:
        from gi.repository import Gtk

        submenu = Gtk.Menu()
        menu_item = Gtk.MenuItem(label=title)
        menu_item.set_submenu(submenu)
        self._menu.append(menu_item)
        self._menu_refs[ref] = submenu
        self._menu_refs[MODEL_ITEMS_REF] = []
        for label, row in choices:
            item = self._append_item(
                submenu,
                label,
                lambda _item, model_id=row.model_id, backend=row.backend: self._after_action(
                    lambda: self._set_model(model_id, backend)
                ),
            )
            self._menu_refs[MODEL_ITEMS_REF].append((item, row))
