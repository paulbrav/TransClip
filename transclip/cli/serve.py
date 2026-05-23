from __future__ import annotations

from transclip.service import run_server
from transclip.settings import Settings


def handle_serve(settings: Settings) -> int:
    run_server(settings)
    return 0
