from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .settings import Settings
from .text_generation import TextGenerationBackend
from .timing import timed_ms

MODEL_CLEANUP_SYSTEM_PROMPT = (
    "Clean this ASR transcript faithfully. Preserve meaning. Correct punctuation, "
    "capitalization, and paragraphing. Remove filler only when it is clearly filler. "
    "Preserve technical terms, flags, identifiers, paths, and code-like text. "
    "Do not add facts. Output only the cleaned transcript."
)


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
class CleanupPromptPolicy:
    system_prompt: str
    max_context_tokens: int = 512
    min_context_tokens: int = 64
    tokens_per_word: int = 3

    def token_budget(self, text: str) -> int:
        return max(self.min_context_tokens, min(self.max_context_tokens, len(text.split()) * self.tokens_per_word))

    def messages(self, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Transcript:\n{text}"},
        ]

    def validate_output(self, cleaned: str, provider: str) -> str:
        cleaned = cleaned.strip()
        if not cleaned:
            raise RuntimeError(f"{provider} cleanup produced an empty response")
        return cleaned


MODEL_CLEANUP_POLICY = CleanupPromptPolicy(MODEL_CLEANUP_SYSTEM_PROMPT)


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    dictation_mode: Literal["rule", "model"]
    requires_text_model: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> CleanupPlan:
        uses_model = settings.text_model_runtime == "transformers" and settings.voice_model_cleanup_always_on
        requires_text_model = settings.text_model_runtime == "transformers" and (
            settings.voice_mode_routing_enabled or uses_model
        )
        return cls(
            dictation_mode="model" if uses_model else "rule",
            requires_text_model=requires_text_model,
        )

    def backend_label(self, *, rule_name: str, text_backend: str, text_model: str) -> str:
        if self.dictation_mode == "model":
            return f"{text_backend}:{text_model}"
        return rule_name


class ModelCleanupProcessor:
    def __init__(self, text_backend: TextGenerationBackend):
        self.text_backend = text_backend
        self.policy = MODEL_CLEANUP_POLICY

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


def apply_dictation_cleanup(
    text: str,
    plan: CleanupPlan,
    *,
    rule_cleanup: CleanupBackend,
    model_cleanup: ModelCleanupProcessor,
) -> CleanupResult:
    if plan.dictation_mode == "model":
        return model_cleanup.cleanup(text)
    return rule_cleanup.cleanup(text)


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
