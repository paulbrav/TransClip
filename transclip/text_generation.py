from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from .device import resolve_torch_device
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


def build_text_generation_backend(settings: Settings) -> TextGenerationBackend:
    runtime = settings.text_model_runtime.lower()
    if runtime == "transformers":
        return TransformersTextGenerationBackend(
            settings.text_model,
            local_files_only=settings.models_local_files_only,
            cache_dir=settings.model_cache_dir,
        )
    raise ValueError(f"Unsupported text model runtime: {settings.text_model_runtime}")
