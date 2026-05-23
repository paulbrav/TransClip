from __future__ import annotations

from pathlib import Path

from .platform_runtime import get_runtime
from .settings import Settings
from .tray_gtk import run_python_tray


def run_tray(settings: Settings, explicit_settings_path: Path | None = None) -> int:
    runtime = get_runtime()
    if runtime.system() == "Darwin":
        from .tray_macos import run_macos_tray

        return run_macos_tray(settings, explicit_settings_path=explicit_settings_path, runtime=runtime)
    if runtime.system() == "Windows":
        from .tray_win32 import run_windows_tray

        return run_windows_tray(settings, explicit_settings_path=explicit_settings_path, runtime=runtime)
    return run_python_tray(settings, explicit_settings_path=explicit_settings_path)
