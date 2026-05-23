from __future__ import annotations

import os
import platform as py_platform

from .cleanup import CleanupPlan
from .device import resolve_torch_device, torch_cuda_usable, torch_mps_available
from .doctor_platform import check_macos_version
from .doctor_types import Check
from .models import (
    cache_artifacts_present,
    model_cache_path,
    model_cache_root,
    normalize_asr_backend,
    required_model_cache_paths,
    resolve_catalog_entry,
)
from .platform_runtime import PlatformRuntime, get_runtime
from .runtime_profile import detect_runtime_profile, is_apple_silicon, is_native_arm_python
from .settings import Settings


def build_backend_checks(settings: Settings, runtime: PlatformRuntime | None = None) -> list[Check]:
    if settings.asr_backend.startswith("file:"):
        profile = detect_runtime_profile(runtime)
        if settings.asr_backend == "file:/dev/null" and profile.profile_id in {"darwin_other", "unsupported"}:
            return [
                check_model_cache(settings, runtime),
                Check(
                    "asr_config",
                    False,
                    f"unsupported platform for production ASR: {profile.system} {profile.architecture}",
                ),
            ]
        return [
            check_model_cache(settings, runtime),
            Check("asr_runtime", True, "file backend has no extra runtime checks", required=True),
        ]
    try:
        backend = normalize_asr_backend(settings.asr_backend)
        entry = resolve_catalog_entry(settings, runtime)
    except ValueError as exc:
        return [Check("asr_config", False, str(exc))]
    checks = [check_model_cache(settings, runtime)]
    if entry is not None and entry.runtime_kind == "mlx":
        checks.extend(build_mlx_checks(settings, runtime))
    elif backend == "granite_nar":
        checks.extend(
            [
                check_torch_runtime(settings, runtime),
                check_asr_runtime(settings),
            ]
        )
    elif entry is not None and entry.runtime_kind == "torch":
        checks.append(check_torch_runtime(settings, runtime))
    else:
        checks.append(Check("asr_runtime", True, f"{settings.asr_backend} has no extra runtime checks"))
    return checks


def build_mlx_checks(settings: Settings, runtime: PlatformRuntime | None = None) -> list[Check]:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    return [
        Check(
            "mlx_platform",
            system == "Darwin",
            "MLX ASR requires macOS" if system != "Darwin" else "macOS",
        ),
        Check(
            "mlx_apple_silicon",
            is_apple_silicon(runtime),
            "MLX ASR requires Apple Silicon (arm64)"
            if not is_apple_silicon(runtime)
            else f"architecture={detect_runtime_profile(runtime).architecture}",
        ),
        Check(
            "mlx_native_python",
            is_native_arm_python(),
            f"Python architecture is {py_platform.machine()}; MLX requires native ARM Python",
        ),
        check_macos_version(runtime),
        check_mlx_import(),
        check_mlx_audio_import(),
        check_mlx_model_cache(settings, runtime),
    ]


def check_model_cache(settings: Settings, runtime: PlatformRuntime | None = None) -> Check:
    cache_root = model_cache_root(settings, runtime)
    extra_model_ids = _extra_text_model_ids(settings)
    required_paths = required_model_cache_paths(
        settings,
        extra_model_ids=extra_model_ids,
        runtime=runtime,
    )
    if not required_paths:
        return Check("model_cache", True, "no local model artifacts required")
    missing = [str(path) for path in required_paths if not path.exists()]
    if not missing:
        return Check("model_cache", True, f"found model artifacts under {cache_root}")
    commands = _prefetch_commands_for_missing_cache(settings, extra_model_ids, runtime)
    remediation = "; run: " + "; ".join(commands) if commands else ""
    return Check(
        "model_cache",
        False,
        "missing local model artifacts: " + ", ".join(missing) + remediation,
    )


def check_mlx_model_cache(settings: Settings, runtime: PlatformRuntime | None = None) -> Check:
    if cache_artifacts_present(settings.asr_model, settings, runtime):
        return Check("mlx_model_cache", True, f"found MLX snapshot for {settings.asr_model}")
    return Check(
        "mlx_model_cache",
        False,
        f"missing MLX model cache for {settings.asr_model}; "
        f"run: transclip models prefetch --model {settings.asr_model}",
    )


def check_torch_runtime(settings: Settings, runtime: PlatformRuntime | None = None) -> Check:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    try:
        import torch
    except ImportError:
        return Check("torch_runtime", False, "torch is not installed; install transclip[models]")
    requested = settings.asr_device.lower()
    cuda_usable = torch_cuda_usable()
    mps_usable = torch_mps_available()
    version = getattr(torch, "__version__", "unknown")
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if requested in {"cuda", "rocm"} and not cuda_usable:
        if system == "Windows":
            detail = f"torch {version}; requested {settings.asr_device}, but CUDA tensor smoke failed"
        else:
            detail = (
                f"torch {version} hip={hip}; requested {settings.asr_device}, but GPU tensor smoke failed"
            )
        return Check("torch_runtime", False, detail)
    if requested == "mps" and not mps_usable:
        return Check("torch_runtime", False, f"torch {version}; requested MPS, but MPS is unavailable")
    if cuda_usable:
        if system == "Windows":
            return Check("torch_runtime", True, f"torch {version}; CUDA tensor smoke passed")
        return Check("torch_runtime", True, f"torch {version} hip={hip}; GPU tensor smoke passed")
    if mps_usable:
        return Check("torch_runtime", True, f"torch {version}; MPS available")
    if system == "Windows":
        return Check("torch_runtime", True, f"torch {version}; auto will use CPU")
    return Check("torch_runtime", True, f"torch {version} hip={hip}; auto will use CPU")


def check_asr_runtime(settings: Settings) -> Check:
    if normalize_asr_backend(settings.asr_backend) != "granite_nar":
        return Check("asr_runtime", True, f"{settings.asr_backend} has no extra runtime checks")
    try:
        import torch
    except ImportError:
        return Check("asr_runtime", False, "torch is not installed")
    device = resolve_torch_device(settings.asr_device)
    if device != "cuda":
        return Check(
            "asr_runtime",
            False,
            "Granite NAR requires CUDA/ROCm with flash-attn; use asr_backend='granite' for CPU",
        )
    if getattr(torch.version, "hip", None):
        os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        return Check(
            "asr_runtime",
            False,
            f"Granite NAR requires flash-attn (project-validated ROCm where available); import failed: {exc}",
        )
    return Check(
        "asr_runtime",
        True,
        "Granite NAR flash-attn runtime import passed (ROCm support is project-validated, not an IBM guarantee)",
    )


def check_mlx_import() -> Check:
    try:
        import mlx  # noqa: F401
    except ImportError as exc:
        return Check("mlx_import", False, f"mlx import failed: {exc}; install transclip[mlx]")
    return Check("mlx_import", True, "mlx import passed")


def check_mlx_audio_import() -> Check:
    try:
        import mlx_audio  # noqa: F401
    except ImportError as exc:
        return Check("mlx_audio_import", False, f"mlx_audio import failed: {exc}; install transclip[mlx]")
    return Check("mlx_audio_import", True, "mlx_audio import passed")


def _extra_text_model_ids(settings: Settings) -> tuple[str, ...]:
    if CleanupPlan.from_settings(settings).requires_text_model:
        return (settings.text_model,)
    return ()


def _prefetch_commands_for_missing_cache(
    settings: Settings,
    extra_model_ids: tuple[str, ...],
    runtime: PlatformRuntime | None = None,
) -> list[str]:
    model_ids = [settings.asr_model] if settings.asr_model and not settings.asr_backend.startswith("file:") else []
    model_ids.extend(extra_model_ids)

    commands = []
    seen = set()
    for model_id in model_ids:
        if not model_id or model_id in seen or model_cache_path(model_id, settings, runtime).exists():
            continue
        seen.add(model_id)
        commands.append(f"transclip models prefetch --model {model_id}")
    return commands
