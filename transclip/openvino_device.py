from __future__ import annotations

from functools import lru_cache

OpenVINODevice = str

_OPENVINO_PREFIX = "openvino:"


@lru_cache(maxsize=1)
def openvino_available_devices() -> tuple[str, ...]:
    """Return the OpenVINO devices on this host (e.g. CPU, GPU.0, NPU).

    Import-safe: returns an empty tuple when OpenVINO is not installed or device
    enumeration fails, so callers (profile detection, doctor) never raise.
    Enumeration is cheap and does not allocate device memory, so unlike the CUDA
    smoke test there is no subprocess guard.
    """
    try:
        import openvino

        return tuple(openvino.Core().available_devices)
    except Exception:
        return ()


def download_openvino_snapshot(repo_id: str, cache_dir: str = "") -> str:
    """Download a Hugging Face snapshot for an OpenVINO model, returning its path.

    Shared by the OpenVINO ASR and text-generation backends so the import guard
    and download call live in one place.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch OpenVINO models. Install transclip[openvino]."
        ) from exc
    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir or None)


def has_intel_accelerator() -> bool:
    """True when an Intel iGPU or NPU is available via OpenVINO.

    Standard OpenVINO only exposes Intel CPU/GPU/NPU plugins, so any GPU/NPU
    entry implies Intel acceleration. NVIDIA GPUs are not enumerated here.
    """
    return any(_is_accelerator(device) for device in openvino_available_devices())


def resolve_openvino_device(requested: str = "auto") -> OpenVINODevice:
    """Translate a settings device value into an OpenVINO device string.

    Accepts ``auto`` and the namespaced forms ``openvino:AUTO|GPU|NPU|CPU``
    (and indexed GPUs like ``openvino:GPU.1``). ``auto`` maps to OpenVINO's
    ``AUTO`` plugin, which picks NPU/GPU/CPU itself. Explicit ``GPU``/``NPU``
    are verified against the available devices; ``CPU`` is always allowed (the
    universal plugin used for testing without Intel hardware).
    """
    value = (requested or "auto").strip()
    if value.lower().startswith(_OPENVINO_PREFIX):
        value = value[len(_OPENVINO_PREFIX) :].strip()
    normalized = value.upper()
    if normalized in {"", "AUTO"}:
        return "AUTO"
    if normalized == "CPU":
        return "CPU"
    if normalized in {"NPU", "GPU"} or normalized.startswith("GPU."):
        available = openvino_available_devices()
        if not _device_present(normalized, available):
            listed = ", ".join(available) if available else "none"
            raise RuntimeError(
                f"OpenVINO device {normalized!r} was requested, but available devices are: {listed}"
            )
        return normalized
    raise ValueError(
        f"Unsupported OpenVINO device: {requested!r}. "
        "Use auto, openvino:AUTO, openvino:GPU, openvino:NPU, or openvino:CPU."
    )


def _is_accelerator(device: str) -> bool:
    return device in {"NPU", "GPU"} or device.startswith("GPU.")


def _device_present(device: str, available: tuple[str, ...]) -> bool:
    if device in available:
        return True
    if device == "GPU":
        return any(entry == "GPU" or entry.startswith("GPU.") for entry in available)
    return False
