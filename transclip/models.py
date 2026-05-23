from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from transclip.platform.profiles import detect_runtime_profile, is_apple_silicon, machine_architecture
from transclip.platform.runtime import PlatformRuntime, get_runtime, user_cache_dir

from .settings import Settings, default_settings

GIB = 1024**3
ModelRuntimeKind = Literal["torch", "mlx", "file"]
PrefetchStrategy = Literal["transformers", "snapshot_download", "none"]


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    model_id: str
    backend: str
    runtime_kind: ModelRuntimeKind
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
        supported_platforms=frozenset({"Darwin", "Linux", "Windows"}),
        supported_architectures=frozenset({"arm64", "aarch64"}),
        dependency_extra="models",
        prefetch_strategy="transformers",
    ),
    ModelCatalogEntry(
        model_id="ibm-granite/granite-speech-4.1-2b-plus",
        backend="granite",
        runtime_kind="torch",
        estimated_bytes=8 * GIB,
        supported_platforms=frozenset({"Darwin", "Linux", "Windows"}),
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

SUPPORTED_MODELS = list(MODEL_CATALOG)


@dataclass(frozen=True, slots=True)
class ModelRow:
    model_id: str
    backend: str
    runtime: str
    marker: str
    cached: bool
    cache_path: str


SUPPORTED_TEXT_MODELS = [
    ModelCatalogEntry(
        model_id="Qwen/Qwen3.5-4B",
        backend="text_generation",
        runtime_kind="torch",
        estimated_bytes=10 * GIB,
        supported_platforms=frozenset({"Darwin", "Linux", "Windows"}),
        supported_architectures=None,
        dependency_extra="models",
        prefetch_strategy="transformers",
    ),
]

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
    for entry in MODEL_CATALOG + tuple(SUPPORTED_TEXT_MODELS):
        if entry.model_id == model_id:
            return entry
    supported = ", ".join(entry.model_id for entry in MODEL_CATALOG + tuple(SUPPORTED_TEXT_MODELS))
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
    platform_runtime_instance = get_runtime(runtime)
    system = platform_runtime_instance.system()
    if system not in entry.supported_platforms:
        raise ValueError(
            f"Model {entry.model_id} is not supported on {system}. "
            f"Supported platforms: {', '.join(sorted(entry.supported_platforms))}"
        )
    if entry.supported_architectures is not None and system == "Darwin":
        arch = machine_architecture(runtime)
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
    if backend == "granite_nar" and profile.granite_nar_unsupported_reason:
        raise ValueError(profile.granite_nar_unsupported_reason)
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


def required_model_cache_paths(
    settings: Settings,
    *,
    extra_model_ids: tuple[str, ...] = (),
    runtime: PlatformRuntime | None = None,
) -> list[Path]:
    paths: list[Path] = []
    if not settings.asr_backend.startswith("file:") and settings.asr_model:
        paths.append(model_cache_path(settings.asr_model, settings, runtime))
    for model_id in extra_model_ids:
        if not model_id:
            continue
        path = model_cache_path(model_id, settings, runtime)
        if path not in paths:
            paths.append(path)
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


def model_rows(settings: Settings, runtime: PlatformRuntime | None = None) -> list[ModelRow]:
    defaults = default_settings(runtime)
    rows: list[ModelRow] = []
    for entry in supported_catalog_entries(runtime):
        markers = []
        if entry.model_id == settings.asr_model and entry.backend == normalize_asr_backend(settings.asr_backend):
            markers.append("current")
        if entry.model_id == defaults.asr_model and entry.backend == normalize_asr_backend(defaults.asr_backend):
            markers.append("default")
        rows.append(
            ModelRow(
                model_id=entry.model_id,
                backend=entry.backend,
                runtime=entry.runtime_kind,
                marker=",".join(markers) if markers else "-",
                cached=cache_artifacts_present(entry.model_id, settings, runtime),
                cache_path=str(model_cache_path(entry.model_id, settings, runtime)),
            )
        )
    for model in SUPPORTED_TEXT_MODELS:
        markers = []
        if model.model_id == settings.text_model:
            markers.append("current-text")
        if model.model_id == defaults.text_model:
            markers.append("default-text")
        rows.append(
            ModelRow(
                model_id=model.model_id,
                backend=model.backend,
                runtime=model.runtime_kind,
                marker=",".join(markers) if markers else "-",
                cached=cache_artifacts_present(model.model_id, settings, runtime),
                cache_path=str(model_cache_path(model.model_id, settings, runtime)),
            )
        )
    return rows


def asr_model_rows(settings: Settings, runtime: PlatformRuntime | None = None) -> list[ModelRow]:
    return [row for row in model_rows(settings, runtime) if row.backend != "text_generation"]


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
    if entry.backend != "text_generation":
        validate_platform_support(entry, runtime)
    ensure_disk_space(settings, entry, runtime)
    cache_dir = str(model_cache_root(settings, runtime))

    if entry.prefetch_strategy == "snapshot_download":
        _prefetch_snapshot(entry, cache_dir)
        return model_cache_path(entry.model_id, settings, runtime)

    if entry.prefetch_strategy == "transformers":
        try:
            handler = _PREFETCH_BY_BACKEND[entry.backend]
        except KeyError as exc:
            raise RuntimeError(
                f"No prefetch handler for backend {entry.backend!r} (model {entry.model_id!r})"
            ) from exc
        handler(entry, cache_dir)
        return model_cache_path(entry.model_id, settings, runtime)

    if entry.prefetch_strategy == "none":
        return model_cache_path(entry.model_id, settings, runtime)

    raise RuntimeError(f"Unsupported prefetch strategy {entry.prefetch_strategy!r} for {entry.model_id!r}")


def _prefetch_snapshot(entry: ModelCatalogEntry, cache_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for MLX model prefetch.") from exc
    snapshot_download(
        repo_id=entry.model_id,
        cache_dir=cache_dir,
    )


def _prefetch_text_generation(entry: ModelCatalogEntry, cache_dir: str) -> None:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers is required. Install transclip[models].") from exc
    AutoModelForImageTextToText.from_pretrained(
        entry.model_id,
        dtype="auto",
        local_files_only=False,
        cache_dir=cache_dir,
    )
    AutoProcessor.from_pretrained(
        entry.model_id,
        local_files_only=False,
        cache_dir=cache_dir,
    )


def _prefetch_granite_nar(entry: ModelCatalogEntry, cache_dir: str) -> None:
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


def _prefetch_transformers_speech(entry: ModelCatalogEntry, cache_dir: str) -> None:
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


_PREFETCH_BY_BACKEND = {
    "text_generation": _prefetch_text_generation,
    "granite_nar": _prefetch_granite_nar,
    "granite": _prefetch_transformers_speech,
}
