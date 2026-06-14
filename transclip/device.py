from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from typing import Literal

TorchDevice = Literal["cpu", "cuda", "mps"]


def resolve_torch_device(requested: str = "auto") -> TorchDevice:
    value = requested.lower()
    if value == "rocm":
        value = "cuda"
    if value not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of auto, cpu, cuda, mps, or rocm")
    if value == "cpu":
        return "cpu"
    if value == "mps":
        if not torch_mps_available():
            raise RuntimeError("MPS was requested, but torch reports it is unavailable")
        return "mps"
    if value == "cuda":
        if not torch_cuda_usable():
            raise RuntimeError("CUDA/ROCm was requested, but torch cannot execute a GPU tensor operation")
        return "cuda"
    if torch_cuda_usable():
        return "cuda"
    if torch_mps_available():
        return "mps"
    return "cpu"


def torch_mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    backends = getattr(torch, "backends", None)
    if backends is None:
        return False
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    is_available = getattr(mps, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


@lru_cache(maxsize=1)
def torch_cuda_usable() -> bool:
    script = """
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
x = torch.randn((64, 64), device="cuda")
y = x @ x
torch.cuda.synchronize()
float(y[0, 0].detach().cpu())
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
