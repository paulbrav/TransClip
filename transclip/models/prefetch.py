from __future__ import annotations

from pathlib import Path

from transclip.platform.runtime import PlatformRuntime
from transclip.settings import Settings

from .cache import ensure_disk_space, model_cache_path, model_cache_root
from .catalog import model_by_id, validate_platform_support
from .types import ModelCatalogEntry


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
