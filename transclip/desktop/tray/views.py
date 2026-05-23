from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class RefDrivenMenuView:
    """Shared TrayMenuView implementation for ref-keyed platform menu widgets."""

    def __init__(
        self,
        menu_refs: dict[str, Any],
        *,
        set_item_label: Callable[[Any, str], None],
        set_item_enabled: Callable[[Any, bool], None],
        rebuild_history: Callable[[Sequence[tuple[str, str]]], None],
        set_health_icon: Callable[[str], None],
    ) -> None:
        self._menu_refs = menu_refs
        self._set_item_label = set_item_label
        self._set_item_enabled = set_item_enabled
        self._rebuild_history = rebuild_history
        self._set_health_icon = set_health_icon

    def set_label(self, ref: str, text: str) -> None:
        self._set_item_label(self._menu_refs[ref], text)

    def set_enabled(self, ref: str, enabled: bool) -> None:
        self._set_item_enabled(self._menu_refs[ref], enabled)

    def set_model_labels(self, rows: Sequence[tuple[Any, str]]) -> None:
        for item, label in rows:
            self._set_item_label(item, label)

    def rebuild_history(self, entries: Sequence[tuple[str, str]]) -> None:
        self._rebuild_history(entries)

    def set_health_icon(self, icon: str) -> None:
        self._set_health_icon(icon)
