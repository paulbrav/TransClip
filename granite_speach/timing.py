from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timed_ms(target: dict[str, float], key: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        target[key] = round((perf_counter() - start) * 1000, 3)
