from __future__ import annotations

from pathlib import Path
import wave

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
            raise RuntimeError("Install granite-speach[audio] for microphone capture.") from exc
        self._np = np
        self._sd = sd
        self._frames = []

        def callback(indata, frames, time, status):
            del frames, time
            if status:
                return
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()

    def stop_to_wav(self, output_path: Path) -> Path:
        if self._stream is None or self._np is None:
            raise RuntimeError("Recorder is not running")
        self._stream.stop()
        self._stream.close()
        audio = self._np.concatenate(self._frames) if self._frames else self._np.zeros((0, 1), dtype="int16")
        write_wav(output_path, audio.tobytes(), self.settings.sample_rate)
        self._stream = None
        return output_path


def write_wav(path: Path, pcm16_mono: bytes, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16_mono)
    return path
