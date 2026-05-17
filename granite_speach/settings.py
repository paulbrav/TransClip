from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import platform
import tomllib


DEFAULT_KEYWORDS = [
    "PyTorch",
    "ROCm",
    "gfx1151",
    "Tauri",
    "llama.cpp",
    "Gemma",
    "Granite",
    "Qwen",
    "Transformers",
    "Hugging Face",
    "MLX",
    "Wayland",
]


@dataclass(slots=True)
class Settings:
    hotkey_linux: str = "<Super><Shift>XF86TouchpadOff"
    hotkey_macos: str = "Option+Space"
    language: str = "en"
    asr_model: str = "ibm-granite/granite-speech-4.1-2b-nar"
    cleanup_model: str = "google/gemma-4-E2B-it"
    cleanup_enabled: bool = True
    cleanup_runtime: str = "rule"
    cleanup_model_path: str = ""
    models_local_files_only: bool = True
    model_cache_dir: str = ""
    restore_clipboard_after_paste: bool = True
    clipboard_restore_delay_ms: int = 500
    max_recording_seconds: int = 60
    min_recording_ms: int = 250
    debug_capture: bool = False
    debug_capture_dir: str = "debug-captures"
    asr_backend: str = "granite_nar"
    asr_device: str = "auto"
    sample_rate: int = 16000
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def active_hotkey(self) -> str:
        return self.hotkey_macos if platform.system() == "Darwin" else self.hotkey_linux

    @property
    def paste_shortcut(self) -> str:
        return "Command+V" if platform.system() == "Darwin" else "Ctrl+V"


def default_config_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "granite-speach"
    return Path.home() / ".config" / "granite-speach"


def settings_path(config_dir: Path | None = None) -> Path:
    return (config_dir or default_config_dir()) / "settings.toml"


def keywords_path(config_dir: Path | None = None) -> Path:
    return (config_dir or default_config_dir()) / "keywords.txt"


def load_settings(path: Path | None = None) -> Settings:
    path = path or settings_path()
    if not path.exists():
        return Settings()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(Settings)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown settings field(s): {', '.join(unknown)}")
    return Settings(**data)


def write_default_settings(path: Path | None = None) -> Path:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    settings = Settings()
    lines = [
        f'hotkey_linux = "{settings.hotkey_linux}"',
        f'hotkey_macos = "{settings.hotkey_macos}"',
        f'language = "{settings.language}"',
        "",
        f'asr_model = "{settings.asr_model}"',
        f'cleanup_model = "{settings.cleanup_model}"',
        f"cleanup_enabled = {str(settings.cleanup_enabled).lower()}",
        f'cleanup_runtime = "{settings.cleanup_runtime}"',
        f'cleanup_model_path = "{settings.cleanup_model_path}"',
        f"models_local_files_only = {str(settings.models_local_files_only).lower()}",
        f'model_cache_dir = "{settings.model_cache_dir}"',
        "",
        f"restore_clipboard_after_paste = {str(settings.restore_clipboard_after_paste).lower()}",
        f"clipboard_restore_delay_ms = {settings.clipboard_restore_delay_ms}",
        "",
        f"max_recording_seconds = {settings.max_recording_seconds}",
        f"min_recording_ms = {settings.min_recording_ms}",
        "",
        f"debug_capture = {str(settings.debug_capture).lower()}",
        f'debug_capture_dir = "{settings.debug_capture_dir}"',
        "",
        f'asr_backend = "{settings.asr_backend}"',
        f'asr_device = "{settings.asr_device}"',
        f"sample_rate = {settings.sample_rate}",
        f'host = "{settings.host}"',
        f"port = {settings.port}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_default_keywords(path: Path | None = None) -> Path:
    path = path or keywords_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("\n".join(DEFAULT_KEYWORDS) + "\n", encoding="utf-8")
    return path
