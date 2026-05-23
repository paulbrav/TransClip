from __future__ import annotations

from pathlib import Path

from transclip.settings import Settings


def handle_tray(settings: Settings, settings_path: Path | None) -> int:
    from transclip.desktop.tray import run_tray

    return run_tray(settings, explicit_settings_path=settings_path)
