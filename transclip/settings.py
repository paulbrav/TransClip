from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, get_type_hints

from .platform_runtime import default_platform_runtime, user_config_dir
from .product import CONFIG_DIR_NAME

DEFAULT_HOTKEY_LINUX = "<Super><Shift>XF86TouchpadOff"


@dataclass(slots=True)
class Settings:
    hotkey_linux: str = DEFAULT_HOTKEY_LINUX
    hotkey_macos: str = "Option+Space"
    language: str = "en"
    asr_model: str = "ibm-granite/granite-speech-4.1-2b-nar"
    cleanup_enabled: bool = True
    voice_mode_routing_enabled: bool = True
    voice_model_cleanup_always_on: bool = False
    voice_mode_shell_enabled: bool = True
    text_model_runtime: str = "transformers"
    text_model: str = "Qwen/Qwen3.5-4B"
    shell_syntax_validation_enabled: bool = True
    shellcheck_enabled: bool = True
    models_local_files_only: bool = True
    model_cache_dir: str = ""
    restore_clipboard_after_paste: bool = False
    clipboard_restore_delay_ms: int = 500
    max_recording_seconds: int = 60
    min_recording_ms: int = 250
    toggle_cooldown_ms: int = 500
    debug_capture: bool = False
    debug_capture_dir: str = "debug-captures"
    asr_backend: str = "granite_nar"
    asr_device: str = "auto"
    sample_rate: int = 16000
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def active_hotkey(self) -> str:
        return self.hotkey_macos if default_platform_runtime.system() == "Darwin" else self.hotkey_linux

    @property
    def paste_shortcut(self) -> str:
        return "Command+V" if default_platform_runtime.system() == "Darwin" else "Ctrl+Shift+V"


def default_config_dir() -> Path:
    return user_config_dir(CONFIG_DIR_NAME)


def settings_path(config_dir: Path | None = None) -> Path:
    return (config_dir or default_config_dir()) / "settings.toml"


def load_settings(path: Path | None = None) -> Settings:
    path = path or settings_path()
    if not path.exists():
        return Settings()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(Settings)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown settings field(s): {', '.join(unknown)}")
    settings = Settings(**data)
    return settings


def settings_field_names() -> list[str]:
    return [field.name for field in fields(Settings)]


def settings_to_toml(settings: Settings) -> str:
    values = asdict(settings)
    groups = [
        ("hotkey_linux", "hotkey_macos", "language"),
        (
            "asr_model",
            "cleanup_enabled",
            "voice_mode_routing_enabled",
            "voice_model_cleanup_always_on",
            "voice_mode_shell_enabled",
            "text_model_runtime",
            "text_model",
            "shell_syntax_validation_enabled",
            "shellcheck_enabled",
            "models_local_files_only",
            "model_cache_dir",
        ),
        ("restore_clipboard_after_paste", "clipboard_restore_delay_ms"),
        ("max_recording_seconds", "min_recording_ms", "toggle_cooldown_ms"),
        ("debug_capture", "debug_capture_dir"),
        ("asr_backend", "asr_device", "sample_rate", "host", "port"),
    ]
    lines: list[str] = []
    for group in groups:
        if lines:
            lines.append("")
        for name in group:
            lines.append(f"{name} = {_toml_scalar(values[name])}")
    lines.append("")
    return "\n".join(lines)


def get_setting(settings: Settings, field_name: str) -> Any:
    if field_name not in settings_field_names():
        raise ValueError(f"Unknown settings field(s): {field_name}")
    return getattr(settings, field_name)


def set_setting(path: Path | None, field_name: str, raw_value: str) -> Settings:
    """Update one settings field from a CLI string value and persist canonical TOML."""
    current = load_settings(path)
    allowed = settings_field_names()
    if field_name not in allowed:
        raise ValueError(f"Unknown settings field(s): {field_name}")
    value = coerce_setting_value(field_name, raw_value)
    updated = replace(current, **{field_name: value})
    write_settings(updated, path)
    return updated


def patch_settings(path: Path | None, **changes) -> Settings:
    """Merge typed settings changes into the on-disk config and return the updated object."""
    resolved = path or settings_path()
    current = load_settings(resolved) if resolved.exists() else Settings()
    updated = replace(current, **changes)
    write_settings(updated, path)
    return updated


def write_settings(settings: Settings, path: Path | None = None) -> Path:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(settings_to_toml(settings), encoding="utf-8")
    return path


def coerce_setting_value(field_name: str, raw_value: str) -> Any:
    type_hints = get_type_hints(Settings)
    expected = type_hints[field_name]
    if expected is bool:
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"{field_name} expects a boolean value")
    if expected is int:
        return int(raw_value)
    if expected is float:
        return float(raw_value)
    if expected is str:
        return raw_value
    raise ValueError(f"{field_name} has unsupported type {expected}")


def write_default_settings(path: Path | None = None) -> Path:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    write_settings(Settings(), path)
    return path


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    return json_escape_string(str(value))


def json_escape_string(value: str) -> str:
    import json

    return json.dumps(value)
