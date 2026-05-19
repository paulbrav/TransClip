from __future__ import annotations

import re
from collections.abc import Iterable

_ALIASES = {
    "flashattention": ("flash attention",),
    "gfx1151": ("gfx 1151", "gfx eleven fifty one"),
    "llama.cpp": ("llama c++", "llama cpp", "llama dot cpp"),
    "macos": ("mac os",),
    "nar": ("n", "nahr", "nara"),
    "pythontray": ("python trend",),
    "qwen": ("clan",),
    "radeon": ("radan",),
    "recenttranscripts": ("recent transcript",),
    "rocm": ("rockham", "rocm", "roc m"),
    "therock": ("the rockham", "the rock"),
    "wtype": ("w type",),
    "xdotool": ("x tool", "x dotool", "x do tool"),
}


def restore_keywords(text: str, keywords: Iterable[str]) -> str:
    restored = text
    for keyword in keywords:
        clean_keyword = " ".join(str(keyword).split())
        if not clean_keyword:
            continue
        if _contains_keyword(restored, clean_keyword):
            continue
        for alias in _aliases_for(clean_keyword):
            restored, count = _replace_alias(restored, alias, clean_keyword)
            if count:
                break
    return restored


def _aliases_for(keyword: str) -> list[str]:
    normalized = _normalize_key(keyword)
    aliases = list(_ALIASES.get(normalized, ()))
    spaced = _spaced_variant(keyword)
    if spaced and spaced.lower() != keyword.lower():
        aliases.append(spaced)
    punctuation_spaced = re.sub(r"[^A-Za-z0-9]+", " ", keyword).strip()
    if punctuation_spaced and punctuation_spaced.lower() != keyword.lower():
        aliases.append(punctuation_spaced)
    return aliases


def _contains_keyword(text: str, keyword: str) -> bool:
    return re.search(_phrase_pattern(keyword), text, flags=re.IGNORECASE) is not None


def _replace_alias(text: str, alias: str, keyword: str) -> tuple[str, int]:
    return re.subn(_phrase_pattern(alias), keyword, text, count=1, flags=re.IGNORECASE)


def _phrase_pattern(value: str) -> str:
    pieces = [re.escape(piece) for piece in re.findall(r"[A-Za-z0-9]+", value)]
    if not pieces:
        return re.escape(value)
    return r"(?<![A-Za-z0-9])" + r"[\s.\-+]*".join(pieces) + r"(?![A-Za-z0-9])"


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "", value.lower())


def _spaced_variant(value: str) -> str:
    with_acronyms = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return re.sub(r"[^A-Za-z0-9]+", " ", with_acronyms).strip()
