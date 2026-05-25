from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from transclip.platform.profiles import detect_runtime_profile
from transclip.platform.runtime import PlatformRuntime, get_runtime, user_config_dir

from .product import CONFIG_DIR_NAME

DEFAULT_HOTKEY_LINUX = "<Super><Shift>XF86TouchpadOff"
DEFAULT_HOTKEY_WINDOWS = "ctrl+shift+space"

TextDeliveryMode = Literal["inject", "clipboard_only"]


@dataclass(slots=True)
class Settings:
    hotkey_linux: str = DEFAULT_HOTKEY_LINUX
    hotkey_macos: str = "Option+Space"
    hotkey_windows: str = DEFAULT_HOTKEY_WINDOWS
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
    text_delivery_mode: TextDeliveryMode = "inject"
    focus_aware_paste: bool = True
    terminal_wm_class_patterns: str = ""
    min_recording_ms: int = 250
    toggle_cooldown_ms: int = 500
    debug_capture: bool = False
    debug_capture_dir: str = "debug-captures"
    asr_backend: str = "granite_nar"
    asr_device: str = "auto"
    sample_rate: int = 16000
    host: str = "127.0.0.1"
    port: int = 8765


def active_hotkey(settings: Settings, runtime: PlatformRuntime | None = None) -> str:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system == "Darwin":
        return settings.hotkey_macos
    if system == "Windows":
        return settings.hotkey_windows
    return settings.hotkey_linux


def default_settings(runtime: PlatformRuntime | None = None) -> Settings:
    profile = detect_runtime_profile(runtime)
    return Settings(
        asr_backend=profile.default_asr_backend,
        asr_model=profile.default_asr_model,
        asr_device=profile.default_asr_device,
    )


def default_config_dir() -> Path:
    return user_config_dir(CONFIG_DIR_NAME)


def settings_path(config_dir: Path | None = None) -> Path:
    return (config_dir or default_config_dir()) / "settings.toml"


def load_settings(path: Path | None = None, runtime: PlatformRuntime | None = None) -> Settings:
    path = path or settings_path()
    if not path.exists():
        return default_settings(runtime)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    data.pop("max_recording_seconds", None)  # removed; tolerate legacy configs
    allowed = {field.name for field in fields(Settings)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown settings field(s): {', '.join(unknown)}")
    settings = Settings(**data)
    if settings.text_delivery_mode not in get_args(TextDeliveryMode):
        raise ValueError(
            f"text_delivery_mode expects {' or '.join(get_args(TextDeliveryMode))}, "
            f"got {settings.text_delivery_mode!r}"
        )
    return settings


def settings_field_names() -> list[str]:
    return [field.name for field in fields(Settings)]


def settings_to_toml(settings: Settings) -> str:
    values = asdict(settings)
    groups = [
        ("hotkey_linux", "hotkey_macos", "hotkey_windows", "language"),
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
        ("text_delivery_mode", "focus_aware_paste", "terminal_wm_class_patterns"),
        ("min_recording_ms", "toggle_cooldown_ms"),
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
    if get_origin(expected) is Literal:
        normalized = raw_value.strip().lower()
        allowed = get_args(expected)
        if normalized not in allowed:
            raise ValueError(f"{field_name} expects {' or '.join(allowed)}")
        return normalized
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


def write_default_settings(path: Path | None = None, runtime: PlatformRuntime | None = None) -> Path:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    write_settings(default_settings(runtime), path)
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
