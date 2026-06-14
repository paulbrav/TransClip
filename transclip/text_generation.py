from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

from .device import resolve_torch_device
from .models import mlx_snapshot_path
from .openvino_device import download_openvino_snapshot, resolve_openvino_device
from .settings import Settings
from .timing import timed_ms


@dataclass(frozen=True, slots=True)
class TextGenerationResult:
    text: str
    timings_ms: dict[str, float]
    backend: str
    model: str


class TextGenerationBackend(Protocol):
    name: str
    model_name: str

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> TextGenerationResult: ...


class TransformersTextGenerationBackend:
    name = "transformers"

    def __init__(self, model_name: str, local_files_only: bool = True, cache_dir: str = ""):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._loaded = None
        self._lock = threading.RLock()

    def _load(self):
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "transformers, torch, and accelerate are required. Install transclip[models]."
                ) from exc
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir or None,
            )
            device = resolve_torch_device("auto")
            model_kwargs = {"device_map": "auto"} if device == "cuda" else {}
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype="auto",
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir or None,
                **model_kwargs,
            )
            model.eval()
            self._loaded = (processor, model)
            return self._loaded

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> TextGenerationResult:
        timings: dict[str, float] = {}
        with self._lock:
            processor, model = self._load()
            with timed_ms(timings, "text_generation"):
                inputs = processor.apply_chat_template(
                    _processor_messages(messages),
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                input_len = inputs["input_ids"].shape[-1]
                try:
                    import torch
                except ImportError:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                else:
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return TextGenerationResult(response.strip(), timings, self.name, self.model_name)


def _processor_messages(messages: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]


class OpenVINOTextGenerationBackend:
    """Text cleanup / shell generation on Intel CPU/iGPU/NPU via OpenVINO GenAI.

    Mirrors the transformers backend's lazy, lock-guarded load but uses
    ``openvino_genai.LLMPipeline`` over a local HF snapshot. Defaults to the
    OpenVINO ``AUTO`` device; large LLMs generally run best on the iGPU.
    """

    name = "openvino"

    def __init__(
        self,
        model_name: str,
        settings: Settings | None = None,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        cache_dir: str = "",
    ):
        self.model_name = model_name
        self.settings = settings
        self.device = device
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._pipeline: Any | None = None
        self._resolved_path: str | None = None
        self._lock = threading.RLock()

    def _model_path(self) -> str:
        if self._resolved_path:
            return self._resolved_path
        settings = self.settings
        if settings is not None:
            snapshot = mlx_snapshot_path(self.model_name, settings)
            if snapshot is not None:
                self._resolved_path = str(snapshot)
                return self._resolved_path
        if self.local_files_only:
            raise RuntimeError(
                f"Local OpenVINO model artifacts missing for {self.model_name}. "
                f"Run: transclip models prefetch --model {self.model_name}"
            )
        self._resolved_path = self._download_snapshot()
        return self._resolved_path

    def _download_snapshot(self) -> str:
        return download_openvino_snapshot(self.model_name, self.cache_dir)

    def _load(self):
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import openvino_genai as ov_genai
            except ImportError as exc:
                raise RuntimeError(
                    "openvino-genai is required for the OpenVINO text backend. Install transclip[openvino]."
                ) from exc
            device = resolve_openvino_device(self.device)
            self._pipeline = ov_genai.LLMPipeline(self._model_path(), device)
            return self._pipeline

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> TextGenerationResult:
        timings: dict[str, float] = {}
        with self._lock:
            pipeline = self._load()
            with timed_ms(timings, "text_generation"):
                prompt = _render_chat_prompt(pipeline, messages)
                output = pipeline.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
                text = _openvino_text(output)
        return TextGenerationResult(text.strip(), timings, self.name, self.model_name)


def _render_chat_prompt(pipeline: Any, messages: list[dict[str, str]]) -> str:
    history = [{"role": message["role"], "content": message["content"]} for message in messages]
    tokenizer = pipeline.get_tokenizer()
    # Suppress Qwen "thinking" traces (matches the transformers backend). Not all
    # OpenVINO tokenizers accept enable_thinking, so fall back when it is rejected.
    try:
        return tokenizer.apply_chat_template(history, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(history, add_generation_prompt=True)


def _openvino_text(output: Any) -> str:
    texts = getattr(output, "texts", None)
    if texts:
        return str(texts[0])
    return str(output)


def build_text_generation_backend(settings: Settings) -> TextGenerationBackend:
    runtime = settings.text_model_runtime.lower()
    if runtime == "transformers":
        return TransformersTextGenerationBackend(
            settings.text_model,
            local_files_only=settings.models_local_files_only,
            cache_dir=settings.model_cache_dir,
        )
    if runtime == "openvino":
        return OpenVINOTextGenerationBackend(
            settings.text_model,
            settings,
            local_files_only=settings.models_local_files_only,
            cache_dir=settings.model_cache_dir,
        )
    raise ValueError(f"Unsupported text model runtime: {settings.text_model_runtime}")
