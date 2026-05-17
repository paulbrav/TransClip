from __future__ import annotations

import re
from dataclasses import dataclass

from .device import resolve_torch_device
from .glossary import preserve_terms_instruction
from .settings import Settings
from .timing import timed_ms


@dataclass(slots=True)
class CleanupResult:
    text: str
    timings_ms: dict[str, float]
    backend: str


class CleanupBackend:
    name = "cleanup"

    def cleanup(self, text: str, keywords: list[str]) -> CleanupResult:
        raise NotImplementedError


class FaithfulRuleCleanupBackend(CleanupBackend):
    name = "rule-based"

    def cleanup(self, text: str, keywords: list[str]) -> CleanupResult:
        timings: dict[str, float] = {}
        with timed_ms(timings, "cleanup"):
            cleaned = conservative_cleanup(text, keywords)
        return CleanupResult(cleaned, timings, self.name)


class LlamaCppCleanupBackend(CleanupBackend):
    name = "llama.cpp"

    def __init__(self, model_path: str, model_name: str):
        if not model_path:
            raise ValueError("cleanup_model_path is required for llama.cpp cleanup")
        model_path = model_path.replace("~", __import__("os").path.expanduser("~"), 1)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed. Install granite-speach[llama].") from exc
        self.model_name = model_name
        self.model_path = model_path
        self.llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)

    def cleanup(self, text: str, keywords: list[str]) -> CleanupResult:
        timings: dict[str, float] = {}
        prompt = (
            "Clean this ASR transcript faithfully. Add only punctuation, capitalization, "
            "and conservative paragraphing. Do not add facts, remove meaning, rewrite tone, "
            "or replace technical identifiers.\n"
            f"{preserve_terms_instruction(keywords)}\n\nTranscript:\n{text}\n\nCleaned:"
        )
        with timed_ms(timings, "cleanup"):
            output = self.llm(
                prompt,
                max_tokens=max(64, min(512, len(text.split()) * 3)),
                temperature=0.0,
                stop=["\n\nTranscript:"],
            )
            cleaned = output["choices"][0]["text"].strip()
        if not cleaned:
            raise RuntimeError("llama.cpp cleanup produced an empty response")
        return CleanupResult(cleaned, timings, self.name)


class GemmaTransformersCleanupBackend(CleanupBackend):
    name = "gemma-transformers"

    def __init__(self, model_name: str, local_files_only: bool = True, cache_dir: str = ""):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._loaded = None

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers, torch, and accelerate are required. Install granite-speach[models]."
            ) from exc
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

    def cleanup(self, text: str, keywords: list[str]) -> CleanupResult:
        timings: dict[str, float] = {}
        processor, model = self._load()
        messages = faithful_cleanup_messages(text, keywords)
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
                max_new_tokens=max(64, min(512, len(text.split()) * 3)),
                do_sample=False,
            )
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            parsed = processor.parse_response(response)
            if isinstance(parsed, dict):
                cleaned = str(parsed.get("content") or parsed.get("text") or "").strip()
            else:
                cleaned = str(parsed).strip()
        if not cleaned:
            raise RuntimeError("Gemma cleanup produced an empty response")
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


def faithful_cleanup_messages(text: str, keywords: list[str]) -> list[dict[str, str]]:
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
            "content": f"{preserve_terms_instruction(keywords)}\n\nTranscript:\n{text}",
        },
    ]


def conservative_cleanup(text: str, keywords: list[str] | None = None) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    cleaned = _fix_spacing_around_punctuation(cleaned)
    cleaned = _restore_keyword_spellings(cleaned, keywords or [])
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


def _restore_keyword_spellings(text: str, keywords: list[str]) -> str:
    restored = text
    for keyword in sorted(keywords, key=lambda value: len(value), reverse=True):
        aliases = _keyword_aliases(keyword)
        for alias in aliases:
            restored = re.sub(
                rf"(?<!\w){alias}(?!\w)",
                keyword,
                restored,
                flags=re.IGNORECASE,
            )
    return restored


def _keyword_aliases(keyword: str) -> list[str]:
    aliases = [re.escape(keyword)]
    compact = re.sub(r"[^A-Za-z0-9]+", "", keyword)
    if compact and compact.lower() != keyword.lower():
        aliases.append(re.escape(compact))
    if "." in keyword:
        aliases.append(re.escape(keyword).replace(r"\.", r"(?:\.|\s+dot\s+|\s+)"))
    split_camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", keyword)
    if split_camel != keyword:
        aliases.append(re.escape(split_camel))
    aliases.extend(_KNOWN_KEYWORD_ALIASES.get(keyword.lower(), []))
    return list(dict.fromkeys(aliases))


_KNOWN_KEYWORD_ALIASES = {
    "rocm": [r"rockham", r"rock\s*m", r"roc\s*m"],
    "gfx1151": [r"gfx\s*1151", r"gfxfx\s*1151", r"g\s*f\s*x\s*1151"],
    "nar": [r"n\s*ar", r"and\s+ar"],
    "macos": [r"mac\s*os"],
    "wtype": [r"w\s*type"],
    "xdotool": [r"x\s*dotool", r"x\s*tool", r"xool"],
    "therock": [r"the\s+rock"],
    "radeon": [r"radion", r"radian"],
    "qwen": [r"qwen", r"q\s*wen", r"qn"],
    "mlx": [r"mlxx", r"m\s*l\s*x"],
    "flashattention": [r"flash\s+attention"],
    "llama.cpp": [r"llama\s+cpp", r"llama\.\s*c(?:pp)?", r"llama\s+dot\s+cpp"],
    "global shortcut": [r"global\s+short\s+get", r"global\s+sh\s+get"],
}
