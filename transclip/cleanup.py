"""Optional transcript clean up utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Cleaner:
    """Punctuation and stylistic clean up for transcripts."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.enabled = bool(cfg.get("cleanup_enable", False))
        if not self.enabled:
            self.punct = None
            self.llm = None
            self.prompt_template = ""
            return

        # Punctuation stage
        self.punct = None
        model_type = cfg.get("cleanup_punct_model", "fastpunct")
        if model_type == "fastpunct":
            try:
                from fastpunct import FastPunct

                self.punct = FastPunct()
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("fastpunct unavailable: %s", exc)
        elif model_type == "silero":
            try:
                import onnxruntime as ort
                from silero_punctuation import punctuate

                session = ort.InferenceSession(Path(cfg.get("silero_model", "")).as_posix())
                self.punct = lambda s: punctuate(session, s)
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("silero punctuation unavailable: %s", exc)

        # LLM stage
        self.llm = None
        self.prompt_template = str(
            cfg.get(
                "cleanup_prompt",
                "Rewrite the following transcript into clear, well-punctuated English.\n\nINPUT:\n{text}\n\nCLEANED:",
            )
        )
        try:
            from llama_cpp import Llama

            model_path = Path(cfg.get("cleanup_llm_model", "")).as_posix()
            if model_path:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=1024,
                    n_threads=int(cfg.get("cleanup_threads", 4)),
                    n_gpu_layers=int(cfg.get("cleanup_gpu_layers", 0)),
                )
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("llama.cpp unavailable: %s", exc)

    def __call__(self, text: str) -> str:
        if not self.enabled:
            return text

        result = text
        if self.punct:
            try:
                result = " ".join(self.punct([result]))
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("punctuation failed: %s", exc)

        if self.llm:
            try:
                prompt = self.prompt_template.format(text=result)
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("invalid cleanup prompt: %s", exc)
                prompt = f"{self.prompt_template}\n{result}"
            try:
                res = self.llm(prompt, max_tokens=256, temperature=0.3)
                result = res["choices"][0]["text"].strip()
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("LLM clean up failed: %s", exc)

        return result
