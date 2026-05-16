from __future__ import annotations

from pathlib import Path

from .settings import DEFAULT_KEYWORDS


def load_keywords(path: Path | None = None) -> list[str]:
    if path is None or not path.exists():
        return list(DEFAULT_KEYWORDS)
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        if term not in terms:
            terms.append(term)
    return terms


def keyword_prompt(terms: list[str]) -> str:
    if not terms:
        return "transcribe the speech with proper punctuation and capitalization."
    joined = ", ".join(terms)
    return f"transcribe the speech to text. Keywords: {joined}"


def preserve_terms_instruction(terms: list[str]) -> str:
    if not terms:
        return "Preserve technical identifiers exactly."
    joined = ", ".join(terms)
    return f"Preserve these terms exactly when present: {joined}."
