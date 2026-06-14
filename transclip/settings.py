from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, get_type_hints

from transclip.platform.profiles import detect_runtime_profile
from transclip.platform.runtime import PlatformRuntime, get_runtime, user_config_dir

from .product import CONFIG_DIR_NAME

DEFAULT_HOTKEY_LINUX = "<Super><Shift>XF86TouchpadOff"
DEFAULT_HOTKEY_WINDOWS = "ctrl+shift+space"


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
    paste_injection_delay_ms: int = 250
    clipboard_restore_delay_ms: int = 500
    min_recording_ms: int = 250
    max_recording_ms: int = 300_000
    toggle_cooldown_ms: int = 500
    recording_notifications: bool = True
    debug_capture: bool = False
    debug_capture_dir: str = "debug-captures"
    asr_backend: str = "granite_nar"
    asr_device: str = "auto"
    audio_input_device: str = ""
    incremental_transcription: bool = False
    incremental_commit_threshold_s: float = 10.0
    warm_bucket_shapes_s: int = 16
    streaming_chunk_ms: int = 500
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


def paste_shortcut(settings: Settings, runtime: PlatformRuntime | None = None) -> str:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system == "Darwin":
        return "Command+V"
    if system == "Windows":
        return "Ctrl+V"
    return "Ctrl+Shift+V"


def default_settings(runtime: PlatformRuntime | None = None) -> Settings:
    profile = detect_runtime_profile(runtime)
    return Settings(
        asr_backend=profile.default_asr_backend,
        asr_model=profile.default_asr_model,
        asr_device=profile.default_asr_device,
        text_model_runtime=profile.default_text_model_runtime,
        text_model=profile.default_text_model,
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
    legacy_max_recording_seconds = data.pop("max_recording_seconds", None)
    if legacy_max_recording_seconds is not None and "max_recording_ms" not in data:
        data["max_recording_ms"] = int(float(legacy_max_recording_seconds) * 1000)
    allowed = {field.name for field in fields(Settings)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown settings field(s) in {path}: {', '.join(unknown)}")
    settings = Settings(**data)
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
        (
            "restore_clipboard_after_paste",
            "paste_injection_delay_ms",
            "clipboard_restore_delay_ms",
        ),
        ("min_recording_ms", "max_recording_ms", "toggle_cooldown_ms", "recording_notifications"),
        ("debug_capture", "debug_capture_dir"),
        (
            "asr_backend",
            "asr_device",
            "audio_input_device",
            "incremental_transcription",
            "incremental_commit_threshold_s",
            "warm_bucket_shapes_s",
            "streaming_chunk_ms",
            "sample_rate",
            "host",
            "port",
        ),
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
