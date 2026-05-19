from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .platform_runtime import PlatformRuntime, get_runtime, user_cache_dir
from .runtime_profile import detect_runtime_profile, is_apple_silicon
from .settings import Settings, default_settings

GIB = 1024**3
RuntimeKind = Literal["torch", "mlx", "file"]
PrefetchStrategy = Literal["transformers", "snapshot_download", "none"]


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    model_id: str
    backend: str
    runtime_kind: RuntimeKind
    estimated_bytes: int
    supported_platforms: frozenset[str]
    supported_architectures: frozenset[str] | None
    dependency_extra: str
    prefetch_strategy: PrefetchStrategy


MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        model_id="ibm-granite/granite-speech-4.1-2b-nar",
        backend="granite_nar",
        runtime_kind="torch",
        estimated_bytes=8 * GIB,
        supported_platforms=frozenset({"Linux"}),
        supported_architectures=None,
        dependency_extra="models",
        prefetch_strategy="transformers",
    ),
    ModelCatalogEntry(
        model_id="ibm-granite/granite-speech-4.1-2b",
        backend="granite",
        runtime_kind="torch",
        estimated_bytes=8 * GIB,
        supported_platforms=frozenset({"Darwin", "Linux"}),
        supported_architectures=frozenset({"arm64", "aarch64"}),
        dependency_extra="models",
        prefetch_strategy="transformers",
    ),
    ModelCatalogEntry(
        model_id="ibm-granite/granite-speech-4.1-2b-plus",
        backend="granite",
        runtime_kind="torch",
        estimated_bytes=8 * GIB,
        supported_platforms=frozenset({"Darwin", "Linux"}),
        supported_architectures=frozenset({"arm64", "aarch64"}),
        dependency_extra="models",
        prefetch_strategy="transformers",
    ),
    ModelCatalogEntry(
        model_id="mlx-community/whisper-large-v3-turbo-asr-fp16",
        backend="mlx_audio_whisper",
        runtime_kind="mlx",
        estimated_bytes=int(1.61 * GIB),
        supported_platforms=frozenset({"Darwin"}),
        supported_architectures=frozenset({"arm64", "aarch64"}),
        dependency_extra="mlx",
        prefetch_strategy="snapshot_download",
    ),
    ModelCatalogEntry(
        model_id="mlx-community/granite-4.0-1b-speech-8bit",
        backend="granite_mlx",
        runtime_kind="mlx",
        estimated_bytes=int(2.9 * GIB),
        supported_platforms=frozenset({"Darwin"}),
        supported_architectures=frozenset({"arm64", "aarch64"}),
        dependency_extra="mlx",
        prefetch_strategy="snapshot_download",
    ),
)

# Backward-compatible alias
SupportedModel = ModelCatalogEntry
SUPPORTED_MODELS = list(MODEL_CATALOG)

ASR_BACKEND_ALIASES = {
    "granite_nar": "granite_nar",
    "granite-nar": "granite_nar",
    "nar": "granite_nar",
    "granite": "granite",
    "transformers": "granite",
    "mlx": "mlx_audio_whisper",
    "mlx_audio_whisper": "mlx_audio_whisper",
    "mlx_audio": "mlx_audio_whisper",
    "mlx_whisper": "mlx_audio_whisper",
    "granite_mlx": "granite_mlx",
}


def catalog_entry_for_model(model_id: str) -> ModelCatalogEntry:
    for entry in MODEL_CATALOG:
        if entry.model_id == model_id:
            return entry
    supported = ", ".join(entry.model_id for entry in MODEL_CATALOG)
    raise ValueError(f"Unsupported model: {model_id}. Supported models: {supported}")


def model_by_id(model_id: str) -> ModelCatalogEntry:
    return catalog_entry_for_model(model_id)


def catalog_entry_for_backend(backend: str, model_id: str) -> ModelCatalogEntry:
    normalized = normalize_asr_backend(backend)
    if normalized == "file":
        raise ValueError("file backends do not use the model catalog")
    entry = catalog_entry_for_model(model_id)
    if entry.backend != normalized:
        raise ValueError(
            f"Model {model_id} requires asr_backend={entry.backend!r}, not {backend!r}"
        )
    return entry


def normalize_asr_backend(asr_backend: str) -> str:
    if asr_backend.startswith("file:"):
        return "file"
    try:
        return ASR_BACKEND_ALIASES[asr_backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported ASR backend: {asr_backend}") from exc


def validate_platform_support(
    entry: ModelCatalogEntry,
    runtime: PlatformRuntime | None = None,
) -> None:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system not in entry.supported_platforms:
        raise ValueError(
            f"Model {entry.model_id} is not supported on {system}. "
            f"Supported platforms: {', '.join(sorted(entry.supported_platforms))}"
        )
    if entry.supported_architectures is not None and system == "Darwin":
        arch = _machine_arch(runtime)
        if arch not in entry.supported_architectures:
            raise ValueError(
                f"Model {entry.model_id} requires {', '.join(sorted(entry.supported_architectures))}, "
                f"but this machine reports {arch}"
            )


def validate_asr_model_backend(
    asr_backend: str,
    model_id: str,
    runtime: PlatformRuntime | None = None,
) -> str:
    backend = normalize_asr_backend(asr_backend)
    if backend == "file":
        return backend
    entry = catalog_entry_for_backend(backend, model_id)
    validate_platform_support(entry, runtime)
    if backend == "granite_nar":
        if "granite-speech" not in model_id or "-nar" not in model_id:
            raise ValueError("Granite NAR ASR requires an ibm-granite granite-speech NAR model")
    elif backend == "granite":
        if "-nar" in model_id:
            raise ValueError('Use asr_backend = "granite_nar" with Granite NAR models')
        if "granite-speech" not in model_id:
            raise ValueError("Granite ASR requires an ibm-granite granite-speech model")
    profile = detect_runtime_profile(runtime)
    if backend == "granite_nar" and profile.profile_id == "darwin_arm_mlx":
        raise ValueError(
            "Granite Speech 4.1 NAR is not supported on macOS. "
            'Set asr_backend = "mlx_audio_whisper" or choose a supported MLX model.'
        )
    return backend


def resolve_catalog_entry(settings: Settings, runtime: PlatformRuntime | None = None) -> ModelCatalogEntry | None:
    if settings.asr_backend.startswith("file:"):
        return None
    backend = validate_asr_model_backend(settings.asr_backend, settings.asr_model, runtime)
    return catalog_entry_for_backend(backend, settings.asr_model)


def model_cache_root(settings: Settings, runtime: PlatformRuntime | None = None) -> Path:
    if settings.model_cache_dir:
        return Path(settings.model_cache_dir).expanduser()
    return user_cache_dir("huggingface", runtime) / "hub"


def hf_cache_dir(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def model_cache_path(model_id: str, settings: Settings, runtime: PlatformRuntime | None = None) -> Path:
    return model_cache_root(settings, runtime) / hf_cache_dir(model_id)


def mlx_snapshot_path(model_id: str, settings: Settings, runtime: PlatformRuntime | None = None) -> Path | None:
    cache_path = model_cache_path(model_id, settings, runtime)
    snapshots = cache_path / "snapshots"
    if not snapshots.exists():
        return None
    ref = cache_path / "refs" / "main"
    if ref.exists():
        snapshot = snapshots / ref.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return snapshot
    candidates = sorted(snapshots.iterdir())
    return candidates[-1] if candidates else None


def required_model_cache_paths(settings: Settings, runtime: PlatformRuntime | None = None) -> list[Path]:
    if settings.asr_backend.startswith("file:"):
        paths: list[Path] = []
    else:
        paths = [model_cache_path(settings.asr_model, settings, runtime)]
    if settings.cleanup_runtime == "llama_cpp":
        paths.append(Path(settings.cleanup_model_path).expanduser())
    elif settings.cleanup_runtime == "transformers":
        paths.append(model_cache_path(settings.cleanup_model, settings, runtime))
    return paths


def cache_artifacts_present(
    model_id: str,
    settings: Settings,
    runtime: PlatformRuntime | None = None,
) -> bool:
    path = model_cache_path(model_id, settings, runtime)
    if not path.exists():
        return False
    snapshots = path / "snapshots"
    if snapshots.exists() and any(snapshots.iterdir()):
        return True
    return any(path.iterdir()) if path.is_dir() else False


def supported_catalog_entries(runtime: PlatformRuntime | None = None) -> list[ModelCatalogEntry]:
    profile = detect_runtime_profile(runtime)
    entries: list[ModelCatalogEntry] = []
    for entry in MODEL_CATALOG:
        if entry.supported_platforms and profile.system not in entry.supported_platforms:
            continue
        if (
            entry.supported_architectures is not None
            and profile.system == "Darwin"
            and not is_apple_silicon(runtime)
        ):
            continue
        entries.append(entry)
    return entries


def model_rows(settings: Settings, runtime: PlatformRuntime | None = None) -> list[dict[str, Any]]:
    defaults = default_settings(runtime)
    rows = []
    for entry in supported_catalog_entries(runtime):
        markers = []
        if entry.model_id == settings.asr_model and entry.backend == normalize_asr_backend(settings.asr_backend):
            markers.append("current")
        if entry.model_id == defaults.asr_model and entry.backend == normalize_asr_backend(defaults.asr_backend):
            markers.append("default")
        rows.append(
            {
                "model_id": entry.model_id,
                "backend": entry.backend,
                "runtime": entry.runtime_kind,
                "marker": ",".join(markers) if markers else "-",
                "cached": cache_artifacts_present(entry.model_id, settings, runtime),
                "cache_path": str(model_cache_path(entry.model_id, settings, runtime)),
            }
        )
    return rows


def ensure_disk_space(
    settings: Settings,
    entry: ModelCatalogEntry,
    runtime: PlatformRuntime | None = None,
) -> None:
    root = model_cache_root(settings, runtime)
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    if usage.free < entry.estimated_bytes:
        needed_gib = entry.estimated_bytes / GIB
        free_gib = usage.free / GIB
        raise RuntimeError(
            f"Not enough free disk space for {entry.model_id}: "
            f"need about {needed_gib:.1f} GiB, found {free_gib:.1f} GiB at {probe}"
        )


def prefetch_model(model_id: str, settings: Settings, runtime: PlatformRuntime | None = None) -> Path:
    entry = model_by_id(model_id)
    validate_platform_support(entry, runtime)
    ensure_disk_space(settings, entry, runtime)
    cache_dir = str(model_cache_root(settings, runtime))

    if entry.prefetch_strategy == "snapshot_download":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for MLX model prefetch.") from exc
        snapshot_download(
            repo_id=entry.model_id,
            cache_dir=cache_dir,
        )
        return model_cache_path(entry.model_id, settings, runtime)

    if entry.backend == "granite_nar":
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required. Install transclip[models].") from exc
        AutoModel.from_pretrained(
            entry.model_id,
            trust_remote_code=True,
            local_files_only=False,
            cache_dir=cache_dir,
        )
        AutoProcessor.from_pretrained(
            entry.model_id,
            trust_remote_code=True,
            local_files_only=False,
            cache_dir=cache_dir,
        )
    else:
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required. Install transclip[models].") from exc
        AutoModelForSpeechSeq2Seq.from_pretrained(
            entry.model_id,
            local_files_only=False,
            cache_dir=cache_dir,
        )
        AutoProcessor.from_pretrained(
            entry.model_id,
            local_files_only=False,
            cache_dir=cache_dir,
        )
    return model_cache_path(entry.model_id, settings, runtime)


def _machine_arch(runtime: PlatformRuntime | None) -> str:
    from .runtime_profile import machine_architecture

    return machine_architecture(runtime)
