from __future__ import annotations

from transclip.platform.profiles import detect_runtime_profile, is_apple_silicon, machine_architecture
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.settings import Settings, default_settings

from .cache import cache_artifacts_present, model_cache_path
from .types import GIB, ModelCatalogEntry, ModelRow

MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        model_id="ibm-granite/granite-speech-4.1-2b-nar",
        backend="granite_nar",
        display_name="Fast local ASR - Granite 4.1 NAR",
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
        display_name="Keyword-biased ASR - Granite 4.1",
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
        display_name="Speaker/timestamp ASR - Granite 4.1 Plus",
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
        display_name="MLX Whisper Turbo",
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
        display_name="MLX Granite Speech",
        runtime_kind="mlx",
        estimated_bytes=int(2.9 * GIB),
        supported_platforms=frozenset({"Darwin"}),
        supported_architectures=frozenset({"arm64", "aarch64"}),
        dependency_extra="mlx",
        prefetch_strategy="snapshot_download",
    ),
)

SUPPORTED_MODELS = list(MODEL_CATALOG)


SUPPORTED_TEXT_MODELS = [
    ModelCatalogEntry(
        model_id="Qwen/Qwen3.5-4B",
        backend="text_generation",
        display_name="Qwen3.5 4B",
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


def model_display_name(model_id: str) -> str:
    return catalog_entry_for_model(model_id).display_name


_ALL_CATALOG_ENTRIES: tuple[ModelCatalogEntry, ...] = MODEL_CATALOG + tuple(SUPPORTED_TEXT_MODELS)


def catalog_entry_for_model(model_id: str) -> ModelCatalogEntry:
    for entry in _ALL_CATALOG_ENTRIES:
        if entry.model_id == model_id:
            return entry
    supported = ", ".join(entry.model_id for entry in _ALL_CATALOG_ENTRIES)
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


def _model_row(
    entry: ModelCatalogEntry,
    settings: Settings,
    runtime: PlatformRuntime | None,
    *,
    markers: list[str],
) -> ModelRow:
    return ModelRow(
        model_id=entry.model_id,
        backend=entry.backend,
        runtime=entry.runtime_kind,
        marker=",".join(markers) if markers else "-",
        cached=cache_artifacts_present(entry.model_id, settings, runtime),
        cache_path=str(model_cache_path(entry.model_id, settings, runtime)),
    )


def _asr_row_markers(
    entry: ModelCatalogEntry,
    settings: Settings,
    defaults: Settings,
) -> list[str]:
    markers: list[str] = []
    if entry.model_id == settings.asr_model and entry.backend == normalize_asr_backend(settings.asr_backend):
        markers.append("current")
    if entry.model_id == defaults.asr_model and entry.backend == normalize_asr_backend(defaults.asr_backend):
        markers.append("default")
    return markers


def _text_row_markers(
    entry: ModelCatalogEntry,
    settings: Settings,
    defaults: Settings,
) -> list[str]:
    markers: list[str] = []
    if entry.model_id == settings.text_model:
        markers.append("current-text")
    if entry.model_id == defaults.text_model:
        markers.append("default-text")
    return markers


def model_rows(settings: Settings, runtime: PlatformRuntime | None = None) -> list[ModelRow]:
    defaults = default_settings(runtime)
    rows: list[ModelRow] = []
    for entry in supported_catalog_entries(runtime):
        rows.append(
            _model_row(
                entry,
                settings,
                runtime,
                markers=_asr_row_markers(entry, settings, defaults),
            )
        )
    for entry in SUPPORTED_TEXT_MODELS:
        rows.append(
            _model_row(
                entry,
                settings,
                runtime,
                markers=_text_row_markers(entry, settings, defaults),
            )
        )
    return rows


def asr_model_rows(settings: Settings, runtime: PlatformRuntime | None = None) -> list[ModelRow]:
    return [row for row in model_rows(settings, runtime) if row.backend != "text_generation"]


