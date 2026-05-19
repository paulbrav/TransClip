from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .platform_runtime import user_cache_dir
from .settings import Settings

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class SupportedModel:
    model_id: str
    backend: str
    estimated_bytes: int


SUPPORTED_MODELS = [
    SupportedModel(
        "ibm-granite/granite-speech-4.1-2b-nar",
        "granite_nar",
        8 * GIB,
    ),
    SupportedModel(
        "ibm-granite/granite-speech-4.1-2b",
        "granite",
        8 * GIB,
    ),
]

ASR_BACKEND_ALIASES = {
    "granite_nar": "granite_nar",
    "granite-nar": "granite_nar",
    "nar": "granite_nar",
    "granite": "granite",
    "transformers": "granite",
}


def model_by_id(model_id: str) -> SupportedModel:
    for model in SUPPORTED_MODELS:
        if model.model_id == model_id:
            return model
    supported = ", ".join(model.model_id for model in SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model: {model_id}. Supported models: {supported}")


def normalize_asr_backend(asr_backend: str) -> str:
    if asr_backend.startswith("file:"):
        return "file"
    try:
        return ASR_BACKEND_ALIASES[asr_backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported ASR backend: {asr_backend}") from exc


def validate_asr_model_backend(asr_backend: str, model_id: str) -> str:
    backend = normalize_asr_backend(asr_backend)
    if backend == "file":
        return backend
    if backend == "granite_nar":
        if "granite-speech" not in model_id or "-nar" not in model_id:
            raise ValueError("Granite NAR ASR requires an ibm-granite granite-speech NAR model")
        return backend
    if "-nar" in model_id:
        raise ValueError('Use asr_backend = "granite_nar" with Granite NAR models')
    if "granite-speech" not in model_id:
        raise ValueError("V1 ASR requires an ibm-granite granite-speech model")
    return backend


def model_cache_root(settings: Settings) -> Path:
    if settings.model_cache_dir:
        return Path(settings.model_cache_dir).expanduser()
    return user_cache_dir("huggingface") / "hub"


def hf_cache_dir(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def model_cache_path(model_id: str, settings: Settings) -> Path:
    return model_cache_root(settings) / hf_cache_dir(model_id)


def required_model_cache_paths(settings: Settings) -> list[Path]:
    paths = [model_cache_path(settings.asr_model, settings)]
    if settings.cleanup_runtime == "llama_cpp":
        paths.append(Path(settings.cleanup_model_path).expanduser())
    elif settings.cleanup_runtime == "transformers":
        paths.append(model_cache_path(settings.cleanup_model, settings))
    return paths


def cache_artifacts_present(model_id: str, settings: Settings) -> bool:
    path = model_cache_path(model_id, settings)
    if not path.exists():
        return False
    snapshots = path / "snapshots"
    if snapshots.exists() and any(snapshots.iterdir()):
        return True
    return any(path.iterdir()) if path.is_dir() else False


def model_rows(settings: Settings) -> list[dict[str, Any]]:
    default = Settings()
    rows = []
    for model in SUPPORTED_MODELS:
        markers = []
        if model.model_id == settings.asr_model:
            markers.append("current")
        if model.model_id == default.asr_model:
            markers.append("default")
        rows.append(
            {
                "model_id": model.model_id,
                "backend": model.backend,
                "marker": ",".join(markers) if markers else "-",
                "cached": cache_artifacts_present(model.model_id, settings),
                "cache_path": str(model_cache_path(model.model_id, settings)),
            }
        )
    return rows


def ensure_disk_space(settings: Settings, model: SupportedModel) -> None:
    root = model_cache_root(settings)
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    if usage.free < model.estimated_bytes:
        needed_gib = model.estimated_bytes / GIB
        free_gib = usage.free / GIB
        raise RuntimeError(
            f"Not enough free disk space for {model.model_id}: "
            f"need about {needed_gib:.1f} GiB, found {free_gib:.1f} GiB at {probe}"
        )


def prefetch_model(model_id: str, settings: Settings) -> Path:
    model = model_by_id(model_id)
    ensure_disk_space(settings, model)
    cache_dir = str(model_cache_root(settings)) if settings.model_cache_dir else None
    if model.backend == "granite_nar":
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required. Install transclip[models].") from exc
        AutoModel.from_pretrained(
            model.model_id,
            trust_remote_code=True,
            local_files_only=False,
            cache_dir=cache_dir,
        )
        AutoFeatureExtractor.from_pretrained(
            model.model_id,
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
            model.model_id,
            local_files_only=False,
            cache_dir=cache_dir,
        )
        AutoProcessor.from_pretrained(
            model.model_id,
            local_files_only=False,
            cache_dir=cache_dir,
        )
    return model_cache_path(model.model_id, settings)
