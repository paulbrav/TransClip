from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GIB = 1024**3
ModelRuntimeKind = Literal["torch", "mlx", "file"]
PrefetchStrategy = Literal["transformers", "snapshot_download", "none"]


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    model_id: str
    backend: str
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
    backend: str
    runtime: ModelRuntimeKind
    marker: str
    cached: bool
    cache_path: str
