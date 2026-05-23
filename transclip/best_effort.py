from __future__ import annotations

from collections.abc import Callable


def best_effort(action: Callable[[], object]) -> str | None:
    try:
        action()
    except Exception as exc:
        return str(exc)
    return None
