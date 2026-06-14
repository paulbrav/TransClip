from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
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
        on_updated: Callable[[], None] | None = None,
    ) -> None:
        self._menu_refs = menu_refs
        self._set_item_label = set_item_label
        self._set_item_enabled = set_item_enabled
        self._rebuild_history = rebuild_history
        self._set_health_icon = set_health_icon
        # Mutating-in-place updates the live menu on GTK/macOS, but pystray menu
        # items are immutable: the win32 adapter updates backing state and must
        # repaint via this hook (icon.update_menu()). Defaults to a no-op.
        self._on_updated = on_updated or (lambda: None)
        self._batching = False

    @contextlib.contextmanager
    def batch(self) -> Iterator[None]:
        """Group several updates into a single repaint.

        A full menu refresh mutates ~6 items; without batching the win32 hook
        would repaint once per item. Inside this context the repaint is deferred
        and fired exactly once on exit.
        """
        outer = self._batching
        self._batching = True
        try:
            yield
        finally:
            self._batching = outer
            if not outer:
                self._on_updated()

    def _repaint(self) -> None:
        if not self._batching:
            self._on_updated()

    def set_label(self, ref: str, text: str) -> None:
        self._set_item_label(self._menu_refs[ref], text)
        self._repaint()

    def set_enabled(self, ref: str, enabled: bool) -> None:
        self._set_item_enabled(self._menu_refs[ref], enabled)
        self._repaint()

    def set_model_labels(self, rows: Sequence[tuple[Any, str]]) -> None:
        for item, label in rows:
            self._set_item_label(item, label)
        self._repaint()

    def rebuild_history(self, entries: Sequence[tuple[str, str]]) -> None:
        self._rebuild_history(entries)
        self._repaint()

    def set_health_icon(self, icon: str) -> None:
        self._set_health_icon(icon)
