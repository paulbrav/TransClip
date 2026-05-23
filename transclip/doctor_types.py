from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True
