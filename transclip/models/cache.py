from __future__ import annotations

import shutil
from pathlib import Path

from transclip.platform.runtime import PlatformRuntime, user_cache_dir
from transclip.settings import Settings

from .types import GIB, ModelCatalogEntry


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
