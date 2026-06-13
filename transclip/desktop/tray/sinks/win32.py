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
        on_copy_history: Callable[[str], None],
    ) -> None:
        self._items = items
        self._menu_refs = menu_refs
        self._pystray = pystray
        self._after_action = after_action
        self._set_model = set_model
        self._on_copy_history = on_copy_history

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

    def _build_history_menu(self) -> Any:
        entries = self._menu_refs.get(
            "_history_entries",
            [("No recent transcripts", "")],
        )
        submenu_items: list = []
        for preview, full_text in entries:
            if not full_text:
                submenu_items.append(self._pystray.MenuItem(preview, None, enabled=False))
                continue
            submenu_items.append(self._pystray.MenuItem(preview, self._copy_history_action(full_text)))
        return self._pystray.Menu(*submenu_items)

    def _copy_history_action(self, value: str) -> Callable[[Any, Any], None]:
        # A factory so `value` is bound here and the handler takes only
        # (icon, item): pystray rejects actions with > 2 positional parameters.
        def handler(_icon: Any, _item: Any) -> None:
            self._on_copy_history(value)

        return handler

    def history_submenu(self, ref: str, title: str, on_open=None) -> None:
        del ref

        def lazy_menu():
            if on_open is not None:
                on_open()
            return self._build_history_menu()

        menu_item = self._pystray.MenuItem(title, self._pystray.Menu(lazy_menu))
        self._items.append(menu_item)
        self._menu_refs["history_menu"] = menu_item

    def model_submenu(self, ref: str, title: str, choices) -> None:
        del ref
        submenu_items: list = []
        self._menu_refs[MODEL_ITEMS_REF] = []
        for label, row in choices:
            model_item = self._pystray.MenuItem(label, self._set_model_action(row.model_id, row.backend))
            submenu_items.append(model_item)
            self._menu_refs[MODEL_ITEMS_REF].append((model_item, row))
        self._items.append(self._pystray.MenuItem(title, self._pystray.Menu(*submenu_items)))

    def _set_model_action(self, model_id: str, backend: str) -> Callable[[Any, Any], None]:
        # Bind the per-row values in this factory scope so the handler takes
        # only (icon, item): pystray rejects actions with > 2 positional args.
        def handler(_icon: Any, _item: Any) -> None:
            self._after_action(lambda: self._set_model(model_id, backend))

        return handler
