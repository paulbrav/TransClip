"""Audio related helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 44100
CHANNELS: int = 1
DTYPE = np.float32


def normalize_and_resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Normalize and resample audio for Whisper."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    if sample_rate != 16000:
        audio = scipy_signal.resample(audio, int(len(audio) * 16000 / sample_rate))
    return audio


def start_input_stream(callback: Any) -> sd.InputStream:
    """Start an audio input stream."""
    default_device = sd.query_devices(kind="input")
    stream = sd.InputStream(
        device=default_device["index"],
        samplerate=int(default_device["default_samplerate"]),
        channels=int(default_device["max_input_channels"]),
        dtype=DTYPE,
        callback=callback,
    )
    stream.start()
    return stream
