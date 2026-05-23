from __future__ import annotations

import time
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

from .settings import Settings


class AudioRecorder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sd = None
        self._np = None
        self._stream = None
        self._frames = []

    def start(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install transclip[audio] for microphone capture.") from exc
        self._np = np
        self._sd = sd
        self._frames = []

        def callback(indata, frames, time, status):
            del frames, time
            if status:
                return
            self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.settings.sample_rate,
                channels=1,
                dtype="int16",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            raise _recording_start_error(exc) from exc

    def stop_to_wav(self, output_path: Path) -> Path:
        audio = self.stop_samples()
        write_wav(output_path, audio.tobytes(), self.settings.sample_rate)
        return output_path

    def stop_samples(self):
        if self._stream is None or self._np is None:
            raise RuntimeError("Recorder is not running")
        self._stream.stop()
        self._stream.close()
        audio = self._np.concatenate(self._frames) if self._frames else self._np.zeros((0, 1), dtype="int16")
        self._stream = None
        return audio


def _recording_start_error(exc: Exception) -> RuntimeError:
    detail = str(exc)
    lowered = detail.lower()
    if "permission" in lowered or "denied" in lowered or "not authorized" in lowered:
        return RuntimeError(
            "Microphone permission denied. On macOS, grant Microphone access in "
            "System Settings > Privacy & Security > Microphone for Terminal, your IDE, "
            "or the LaunchAgent Python process."
        )
    if "no input" in lowered or "device" in lowered:
        return RuntimeError(f"No usable microphone input device: {detail}")
    return RuntimeError(f"Could not start microphone capture: {detail}")


def recording_debug(
    settings: Settings,
    seconds: float = 1.0,
    recorder_cls: type[AudioRecorder] = AudioRecorder,
) -> dict[str, Any]:
    recorder = recorder_cls(settings)
    recorder.start()
    try:
        time.sleep(seconds)
        audio = recorder.stop_samples()
    except Exception:
        with suppress(Exception):
            recorder.stop_samples()
        raise
    frame_count = int(audio.shape[0]) if hasattr(audio, "shape") else 0
    channel_count = int(audio.shape[1]) if hasattr(audio, "shape") and len(audio.shape) > 1 else 1
    duration = frame_count / settings.sample_rate if settings.sample_rate else 0.0
    peak = 0.0
    rms = 0.0
    if frame_count:
        import numpy as np

        audio_float = audio.astype("float32")
        peak = float(np.max(np.abs(audio_float)))
        rms = float(np.sqrt(np.mean(audio_float * audio_float)))
    return {
        "device": sounddevice_summary(),
        "sample_rate": settings.sample_rate,
        "channel_count": channel_count,
        "frame_count": frame_count,
        "duration": duration,
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "silent": peak == 0.0,
    }


def sounddevice_summary() -> str:
    try:
        import sounddevice as sd
    except ImportError:
        return "sounddevice unavailable"
    try:
        default = getattr(sd.default, "device", None)
        input_device = _default_input_device(default)
        if input_device in {None, -1}:
            return f"default input={input_device}"
        info = sd.query_devices(input_device, "input")
        name = info.get("name", "unknown") if isinstance(info, dict) else str(info)
        return f"default input={input_device} {name}"
    except Exception as exc:
        return f"sounddevice query failed: {exc}"


def _default_input_device(default: object) -> object:
    if isinstance(default, tuple):
        return default[0] if default else None
    if isinstance(default, list):
        return default[0] if default else None
    return default


def write_wav(path: Path, pcm16_mono: bytes, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16_mono)
    return path
