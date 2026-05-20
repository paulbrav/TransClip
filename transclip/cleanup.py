from __future__ import annotations

import re
from dataclasses import dataclass

from .device import resolve_torch_device
from .settings import Settings
from .text_generation import TextGenerationBackend
from .timing import timed_ms


@dataclass(slots=True)
class CleanupResult:
    text: str
    timings_ms: dict[str, float]
    backend: str


class CleanupBackend:
    name = "cleanup"

    def cleanup(self, text: str) -> CleanupResult:
        raise NotImplementedError


class FaithfulRuleCleanupBackend(CleanupBackend):
    name = "rule-based"

    def cleanup(self, text: str) -> CleanupResult:
        timings: dict[str, float] = {}
        with timed_ms(timings, "cleanup"):
            cleaned = conservative_cleanup(text)
        return CleanupResult(cleaned, timings, self.name)


@dataclass(frozen=True, slots=True)
class ModelCleanupPolicy:
    max_context_tokens: int = 512
    min_context_tokens: int = 64
    tokens_per_word: int = 3

    def token_budget(self, text: str) -> int:
        return max(self.min_context_tokens, min(self.max_context_tokens, len(text.split()) * self.tokens_per_word))

    def messages(self, text: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Clean this ASR transcript faithfully. Preserve meaning. Correct punctuation, "
                    "capitalization, and paragraphing. Remove filler only when it is clearly filler. "
                    "Preserve technical terms, flags, identifiers, paths, and code-like text. "
                    "Do not add facts. Output only the cleaned transcript."
                ),
            },
            {
                "role": "user",
                "content": f"Transcript:\n{text}",
            },
        ]

    def validate_output(self, cleaned: str, provider: str) -> str:
        cleaned = cleaned.strip()
        if not cleaned:
            raise RuntimeError(f"{provider} cleanup produced an empty response")
        return cleaned


class ModelCleanupProcessor:
    def __init__(self, text_backend: TextGenerationBackend):
        self.text_backend = text_backend
        self.policy = ModelCleanupPolicy()

    def cleanup(self, text: str) -> CleanupResult:
        result = self.text_backend.generate(
            self.policy.messages(text),
            max_new_tokens=self.policy.token_budget(text),
        )
        cleaned = self.policy.validate_output(result.text, result.backend)
        return CleanupResult(
            cleaned,
            dict(result.timings_ms),
            f"{result.backend}:{result.model}",
        )


@dataclass(frozen=True, slots=True)
class FaithfulCleanupPolicy:
    max_context_tokens: int = 512
    min_context_tokens: int = 64
    tokens_per_word: int = 3

    def token_budget(self, text: str) -> int:
        return max(self.min_context_tokens, min(self.max_context_tokens, len(text.split()) * self.tokens_per_word))

    def prompt(self, text: str) -> str:
        return (
            "Clean this ASR transcript faithfully. Add only punctuation, capitalization, "
            "and conservative paragraphing. Do not add facts, remove meaning, rewrite tone, "
            f"or replace technical identifiers.\n\nTranscript:\n{text}\n\nCleaned:"
        )

    def messages(self, text: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You clean ASR transcripts faithfully. Add only punctuation, capitalization, "
                    "and conservative paragraphing. Do not add facts, remove meaning, rewrite tone, "
                    "or replace technical identifiers. Output only the cleaned transcript."
                ),
            },
            {
                "role": "user",
                "content": f"Transcript:\n{text}",
            },
        ]

    def validate_output(self, cleaned: str, provider: str) -> str:
        cleaned = cleaned.strip()
        if not cleaned:
            raise RuntimeError(f"{provider} cleanup produced an empty response")
        return cleaned


class LlamaCppCleanupBackend(CleanupBackend):
    name = "llama.cpp"

    def __init__(self, model_path: str, model_name: str):
        if not model_path:
            raise ValueError("cleanup_model_path is required for llama.cpp cleanup")
        model_path = model_path.replace("~", __import__("os").path.expanduser("~"), 1)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed. Install transclip[llama].") from exc
        self.model_name = model_name
        self.model_path = model_path
        self.llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)
        self.policy = FaithfulCleanupPolicy()

    def cleanup(self, text: str) -> CleanupResult:
        timings: dict[str, float] = {}
        prompt = self.policy.prompt(text)
        with timed_ms(timings, "cleanup"):
            output = self.llm(
                prompt,
                max_tokens=self.policy.token_budget(text),
                temperature=0.0,
                stop=["\n\nTranscript:"],
            )
            cleaned = self.policy.validate_output(output["choices"][0]["text"], self.name)
        return CleanupResult(cleaned, timings, self.name)


class GemmaTransformersCleanupBackend(CleanupBackend):
    name = "gemma-transformers"

    def __init__(self, model_name: str, local_files_only: bool = True, cache_dir: str = ""):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._loaded = None
        self.policy = FaithfulCleanupPolicy()

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers, torch, and accelerate are required. Install transclip[models].") from exc
        processor = AutoProcessor.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir or None,
        )
        device = resolve_torch_device("auto")
        model_kwargs = {"device_map": "auto"} if device == "cuda" else {}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto",
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir or None,
            **model_kwargs,
        )
        model.eval()
        self._loaded = (processor, model)
        return self._loaded

    def cleanup(self, text: str) -> CleanupResult:
        timings: dict[str, float] = {}
        processor, model = self._load()
        messages = self.policy.messages(text)
        with timed_ms(timings, "cleanup"):
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.policy.token_budget(text),
                do_sample=False,
            )
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            parsed = processor.parse_response(response)
            if isinstance(parsed, dict):
                cleaned = str(parsed.get("content") or parsed.get("text") or "").strip()
            else:
                cleaned = str(parsed).strip()
        cleaned = self.policy.validate_output(cleaned, "Gemma")
        return CleanupResult(cleaned, timings, self.name)


def build_cleanup_backend(settings: Settings) -> CleanupBackend:
    runtime = settings.cleanup_runtime.lower()
    if runtime == "llama_cpp":
        return LlamaCppCleanupBackend(settings.cleanup_model_path, settings.cleanup_model)
    if runtime == "transformers":
        return GemmaTransformersCleanupBackend(
            settings.cleanup_model,
            local_files_only=settings.models_local_files_only,
            cache_dir=settings.model_cache_dir,
        )
    if runtime in {"rule", "test_rule"}:
        return FaithfulRuleCleanupBackend()
    raise ValueError(f"Unsupported cleanup runtime: {settings.cleanup_runtime}")


def faithful_cleanup_messages(text: str) -> list[dict[str, str]]:
    return FaithfulCleanupPolicy().messages(text)


def model_cleanup_messages(text: str) -> list[dict[str, str]]:
    return ModelCleanupPolicy().messages(text)


def conservative_cleanup(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    cleaned = _fix_spacing_around_punctuation(cleaned)
    cleaned = _capitalize_sentence_starts(cleaned)
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _fix_spacing_around_punctuation(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", text)
    return text


def _capitalize_sentence_starts(text: str) -> str:
    chars = list(text)
    capitalize_next = True
    for i, char in enumerate(chars):
        if not char.isalpha():
            if char in ".!?":
                next_char = chars[i + 1] if i + 1 < len(chars) else ""
                capitalize_next = not next_char or next_char.isspace()
            continue
        if capitalize_next:
            chars[i] = char.upper()
            capitalize_next = False
    return "".join(chars)
