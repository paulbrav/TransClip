from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


def generate_transcription(model_path: str, audio_path: Path, output_stem: str) -> Any:
    try:
        from mlx_audio.stt.generate import generate_transcription as mlx_generate
    except ImportError as exc:
        raise RuntimeError(
            "mlx-audio is required on macOS Apple Silicon. Install transclip[mlx]."
        ) from exc

    kwargs = {
        "output_path": output_stem,
        "format": "txt",
    }
    signature = inspect.signature(mlx_generate)
    if "model" in signature.parameters:
        kwargs["model"] = model_path
        kwargs["audio"] = str(audio_path)
    else:
        kwargs["model_path"] = model_path
        kwargs["audio_path"] = str(audio_path)
    return mlx_generate(**kwargs)
