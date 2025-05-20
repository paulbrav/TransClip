"""Transcription utilities and model enum."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Dict

import numpy as np
from faster_whisper import WhisperModel
from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class WhisperModelType(StrEnum):
    """Available Whisper model types."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    PARAKEET_TDT_0_6B_V2 = "nvidia/parakeet-tdt-0.6b-v2"

    @classmethod
    def get_description(cls, model_type: "WhisperModelType") -> str:
        """Return a human friendly description for a model type."""
        descriptions: Dict[WhisperModelType, str] = {
            cls.TINY: "Tiny (OpenAI, 39M parameters)",
            cls.BASE: "Base (OpenAI, 74M parameters)",
            cls.SMALL: "Small (OpenAI, 244M parameters)",
            cls.MEDIUM: "Medium (OpenAI, 769M parameters)",
            cls.LARGE: "Large (OpenAI, 1.5B parameters)",
            cls.LARGE_V2: "Large-v2 (OpenAI, 1.5B parameters, improved)",
            cls.LARGE_V3: "Large-v3 (OpenAI, 1.5B parameters, latest)",
            cls.PARAKEET_TDT_0_6B_V2: "Parakeet TDT 0.6B v2 (NVIDIA)",
        }
        return descriptions[model_type]


def get_model_path(model_type: WhisperModelType) -> str:
    """Return a local model path if available."""
    cache_dir = Path.home() / ".cache" / "whisper"
    model_dir = cache_dir / model_type
    return str(model_dir) if model_dir.exists() else model_type


DEFAULT_MODEL_TYPE = WhisperModelType.BASE


class TranscriptionWorker(QThread):
    """Worker thread running Whisper transcription."""

    finished = pyqtSignal(str)

    def __init__(self, audio_data: np.ndarray, model: WhisperModel) -> None:
        super().__init__()
        self.audio_data = audio_data
        self.model = model

    def run(self) -> None:  # pragma: no cover - heavy external call
        """Run the transcription process."""
        try:
            segments, _info = self.model.transcribe(
                self.audio_data,
                language="en",
                beam_size=5,
                initial_prompt="The following is a transcription of spoken English:",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            text = " ".join([segment.text for segment in segments])
            self.finished.emit(text.strip())
        except Exception as exc:  # pragma: no cover - logging
            logger.error("Transcription failed: %s", exc)
            self.finished.emit("")
