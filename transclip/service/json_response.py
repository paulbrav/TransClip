from __future__ import annotations

from typing import TypeVar, cast

T = TypeVar("T")


def json_object_response(payload: object) -> T:
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object, got {type(payload).__name__}")
    return cast(T, payload)
