from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed_ms(target: dict[str, float], key: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        target[key] = round((perf_counter() - start) * 1000, 3)
