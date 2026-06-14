from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GIB = 1024**3
ModelRuntimeKind = Literal["torch", "mlx", "file", "openvino"]
PrefetchStrategy = Literal["transformers", "snapshot_download", "none"]

# Normalized ASR backend kinds produced by normalize_asr_backend /
# validate_asr_model_backend (catalog.py). "file" is the normalized kind for
# any "file:..." backend; it is not stored on a catalog entry.
ASRBackendKind = Literal[
    "granite",
    "granite_nar",
    "granite_mlx",
    "granite_nar_mlx",
    "mlx_audio_whisper",
    "openvino_whisper",
    "file",
]
# The exact backend strings stored on a ModelCatalogEntry. ASR entries reuse the
# ASR backend kinds (minus "file", which catalog entries never hold); text-model
# entries use "text_generation".
CatalogBackendKind = Literal[
    "granite",
    "granite_nar",
    "granite_mlx",
    "granite_nar_mlx",
    "mlx_audio_whisper",
    "openvino_whisper",
    "text_generation",
]


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    model_id: str
    backend: CatalogBackendKind
    display_name: str
    runtime_kind: ModelRuntimeKind
    estimated_bytes: int
    supported_platforms: frozenset[str]
    supported_architectures: frozenset[str] | None
    dependency_extra: str
    prefetch_strategy: PrefetchStrategy


@dataclass(frozen=True, slots=True)
class ModelRow:
    model_id: str
    backend: CatalogBackendKind
    runtime: ModelRuntimeKind
    marker: str
    cached: bool
    cache_path: str
