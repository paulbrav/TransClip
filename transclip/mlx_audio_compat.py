from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


def load_model(model_path: str) -> Any:
    try:
        from mlx_audio.stt.generate import load_model as mlx_load_model
    except ImportError as exc:
        raise RuntimeError(
            "mlx-audio is required on macOS Apple Silicon. Install transclip[mlx]."
        ) from exc

    return mlx_load_model(model_path)


def generate_transcription(model: Any, audio_path: Path, output_stem: str, **generate_options: Any) -> Any:
    try:
        from mlx_audio.stt.generate import generate_transcription as mlx_generate
    except ImportError as exc:
        raise RuntimeError(
            "mlx-audio is required on macOS Apple Silicon. Install transclip[mlx]."
        ) from exc

    kwargs = {
        "output_path": output_stem,
        "format": "txt",
        **generate_options,
    }
    signature = inspect.signature(mlx_generate)
    if "model" in signature.parameters:
        kwargs["model"] = _model_with_decode_option_signature(model, generate_options.keys())
        kwargs["audio"] = str(audio_path)
    else:
        if not isinstance(model, str):
            raise RuntimeError("installed mlx-audio generate_transcription does not accept a loaded model")
        kwargs["model_path"] = model
        kwargs["audio_path"] = str(audio_path)
    return mlx_generate(**kwargs)


class _GenerateCallable:
    def __init__(self, generate: Any, signature: inspect.Signature):
        self._generate = generate
        self.__signature__ = signature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._generate(*args, **kwargs)


class _LoadedModelProxy:
    def __init__(self, model: Any, generate_signature: inspect.Signature):
        self._model = model
        self.generate = _GenerateCallable(model.generate, generate_signature)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def _model_with_decode_option_signature(model: Any, option_names: Any) -> Any:
    if isinstance(model, str):
        return model
    generate = getattr(model, "generate", None)
    if not callable(generate):
        return model
    try:
        signature = inspect.signature(generate)
    except (TypeError, ValueError):
        return model
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return model
    missing = [
        str(name)
        for name in option_names
        if str(name).isidentifier() and str(name) not in signature.parameters
    ]
    if not missing:
        return model
    return _LoadedModelProxy(model, _expanded_generate_signature(signature, missing))


def _expanded_generate_signature(
    signature: inspect.Signature,
    option_names: list[str],
) -> inspect.Signature:
    params = list(signature.parameters.values())
    insert_at = next(
        (
            index
            for index, param in enumerate(params)
            if param.kind == inspect.Parameter.VAR_KEYWORD
        ),
        len(params),
    )
    extra = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
        for name in option_names
    ]
    return signature.replace(parameters=[*params[:insert_at], *extra, *params[insert_at:]])
