from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..menu import MODEL_ITEMS_REF


class _ItemState:
    """Mutable label/enabled state behind a pystray MenuItem.

    pystray MenuItems are immutable - ``text`` and ``enabled`` are read-only - so
    dynamic items are backed by callables that re-read this state on each render.
    The RefDrivenMenuView setters mutate this object and the tray repaints via
    ``icon.update_menu()`` (GTK/macOS mutate their widgets directly instead).
    """

    __slots__ = ("enabled", "text")

    def __init__(self, text: str, enabled: bool = True) -> None:
        self.text = text
        self.enabled = enabled


def _label_of(state: _ItemState) -> Callable[[Any], str]:
    return lambda _item: state.text


def _enabled_of(state: _ItemState) -> Callable[[Any], bool]:
    return lambda _item: state.enabled


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
        state = _ItemState(text, enabled=False)
        item = self._pystray.MenuItem(_label_of(state), None, enabled=_enabled_of(state))
        self._items.append(item)
        self._menu_refs[ref] = state

    def action(self, ref: str, label: str, action, *, enabled: bool = True, callback=None) -> None:
        del action
        state = _ItemState(label, enabled=enabled)
        item = self._pystray.MenuItem(_label_of(state), callback, enabled=_enabled_of(state))
        self._items.append(item)
        if ref:
            self._menu_refs[ref] = state

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
            state = _ItemState(label)
            model_item = self._pystray.MenuItem(_label_of(state), self._set_model_action(row.model_id, row.backend))
            submenu_items.append(model_item)
            self._menu_refs[MODEL_ITEMS_REF].append((state, row))
        self._items.append(self._pystray.MenuItem(title, self._pystray.Menu(*submenu_items)))

    def _set_model_action(self, model_id: str, backend: str) -> Callable[[Any, Any], None]:
        # Bind the per-row values in this factory scope so the handler takes
        # only (icon, item): pystray rejects actions with > 2 positional args.
        def handler(_icon: Any, _item: Any) -> None:
            self._after_action(lambda: self._set_model(model_id, backend))

        return handler
