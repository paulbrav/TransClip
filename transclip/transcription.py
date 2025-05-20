"""Transcription utilities and model enum."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict

import numpy as np
from faster_whisper import WhisperModel

try:
    from nemo.collections.asr.models import EncDecRNNTModel
except Exception:  # pragma: no cover - optional dependency
    EncDecRNNTModel = None
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
    # PARAKEET_TDT_0_6B_V2 = "nvidia/parakeet-tdt-0.6b-v2"  # Disabled: unstable on Python 3.11

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
            # cls.PARAKEET_TDT_0_6B_V2: "Parakeet TDT 0.6B v2 (NVIDIA)",
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


class NeMoTranscriptionWorker(QThread):
    """Worker thread running NeMo transcription."""

    finished = pyqtSignal(str)

    def __init__(self, audio_data: np.ndarray, sample_rate: int, model: Any) -> None:
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.model = model

    def run(self) -> None:  # pragma: no cover - heavy external call
        """Run the NeMo transcription process."""
        if EncDecRNNTModel is None:
            logger.error(
                "nemo_toolkit is not installed. Unable to run NeMo transcription."
            )
            self.finished.emit("")
            return

        import os
        import tempfile
        import wave

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    int16_audio = (self.audio_data * 32767).astype(np.int16)
                    wf.writeframes(int16_audio.tobytes())

            result = self.model.transcribe([tmp_path])
            text = result[0] if isinstance(result, list) else result
            self.finished.emit(str(text).strip())
        except Exception as exc:  # pragma: no cover - logging
            logger.error("NeMo transcription failed: %s", exc)
            self.finished.emit("")
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def load_nemo_model(model_type: WhisperModelType) -> Any:
    """Load a NeMo model, downloading if necessary."""
    if EncDecRNNTModel is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "nemo_toolkit is required to load NeMo models. Install with 'pip install nemo_toolkit[asr]'"
        )

    cache_dir = Path.home() / ".cache" / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_file = cache_dir / f"{model_type.value.replace('/', '_')}.nemo"

    try:
        if model_file.exists():
            return EncDecRNNTModel.restore_from(str(model_file))

        model = EncDecRNNTModel.from_pretrained(model_name=model_type.value)
        try:
            model.save_to(str(model_file))
        except Exception:  # pragma: no cover - optional
            logger.warning("Failed to cache NeMo model", exc_info=True)
        return model
    except Exception as exc:  # pragma: no cover - logging
        logger.error("Failed to load NeMo model: %s", exc)
        raise
