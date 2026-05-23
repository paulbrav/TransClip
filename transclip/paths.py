from __future__ import annotations

from pathlib import Path


def service_settings_path(settings_path: Path) -> str:
    expanded = settings_path.expanduser()
    text = str(expanded)
    if expanded.is_absolute() or (len(text) > 1 and text[1] == ":"):
        return text
    return str(expanded.resolve())
