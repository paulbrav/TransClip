from __future__ import annotations

import logging
import math
import platform as py_platform
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from transclip.platform.runtime import PlatformRuntime

from .device import resolve_torch_device
from .mlx_audio_compat import generate_transcription
from .mlx_audio_compat import load_model as load_mlx_model
from .models import (
    mlx_snapshot_path,
    model_cache_path,
    resolve_catalog_entry,
    validate_asr_model_backend,
)
from .settings import Settings
from .timing import timed_ms

logger = logging.getLogger(__name__)

AR_TOKENS_PER_AUDIO_SECOND = 10
AR_MIN_NEW_TOKENS = 200


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    timings_ms: dict[str, float]
    backend: str
    model: str


class ASRBackend(Protocol):
    name: str
    model: str

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult: ...


@dataclass(slots=True)
class PreparedAudio:
    wav: Any
    sample_rate: int


@dataclass(slots=True)
class PreparedPathAudio:
    wav_path: Path
    sample_rate: int
    duration_seconds: float
    padded_duration_seconds: float
    temporary: bool = False


EDGE_SILENCE_TRIM_THRESHOLD = 0.003
EDGE_SILENCE_TRIM_PADDING_SECONDS = 0.2
MLX_SHORT_AUDIO_BUCKET_SECONDS = 1.0
MLX_SHORT_AUDIO_BUCKET_MAX_SECONDS = 12.0
MLX_AUDIO_BUCKET_SECONDS = 4.0
MLX_MIN_AUDIO_SECONDS = 1.0
MLX_WARM_BUCKET_MAX_SECONDS = 12
MLX_TOKENS_PER_AUDIO_SECOND = 6
MLX_SAMPLE_LEN_PADDING_TOKENS = 32
MLX_MIN_SAMPLE_LEN = 48


class AudioLoader:
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def load_samples(self, wav_path: Path) -> tuple[Any, int]:
        import soundfile as sf

        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        return samples, sample_rate

    @staticmethod
    def fold_mono(samples: Any) -> Any:
        if samples.shape[1] == 1:
            return samples[:, 0]
        return samples.mean(axis=1)


class TorchAudioPreparer:
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.loader = AudioLoader(target_sample_rate)

    def prepare(self, wav_path: Path) -> PreparedAudio:
        import torch

        samples, sample_rate = self.loader.load_samples(wav_path)
        wav = torch.from_numpy(samples.T)
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sample_rate, self.target_sample_rate)
        return PreparedAudio(wav=wav, sample_rate=self.target_sample_rate)


class PathAudioPreparer:
    def __init__(
        self,
        target_sample_rate: int = 16000,
        *,
        bucket_seconds: float = 0.0,
        minimum_seconds: float = 0.0,
        short_bucket_seconds: float = 0.0,
        short_bucket_max_seconds: float = 0.0,
    ):
        self.target_sample_rate = target_sample_rate
        self.bucket_seconds = bucket_seconds
        self.minimum_seconds = minimum_seconds
        self.short_bucket_seconds = short_bucket_seconds
        self.short_bucket_max_seconds = short_bucket_max_seconds
        self.loader = AudioLoader(target_sample_rate)

    def prepare(self, wav_path: Path) -> PreparedPathAudio:
        samples, sample_rate = self.loader.load_samples(wav_path)
        mono = self.loader.fold_mono(samples)
        changed = samples.shape[1] != 1 or sample_rate != self.target_sample_rate
        if sample_rate != self.target_sample_rate:
            mono = _linear_resample(mono, sample_rate, self.target_sample_rate)
        mono, trimmed = _trim_edge_silence(mono, self.target_sample_rate)
        duration_seconds = len(mono) / self.target_sample_rate
        mono, bucket_padded = _pad_to_audio_bucket(
            mono,
            self.target_sample_rate,
            bucket_seconds=self.bucket_seconds,
            minimum_seconds=self.minimum_seconds,
            short_bucket_seconds=self.short_bucket_seconds,
            short_bucket_max_seconds=self.short_bucket_max_seconds,
        )
        padded_duration_seconds = len(mono) / self.target_sample_rate
        changed = changed or bucket_padded
        if not changed and not trimmed:
            return PreparedPathAudio(
                wav_path=wav_path,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                padded_duration_seconds=padded_duration_seconds,
            )

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = Path(handle.name)
        sf.write(str(output), mono, self.target_sample_rate)
        return PreparedPathAudio(
            wav_path=output,
            sample_rate=self.target_sample_rate,
            duration_seconds=duration_seconds,
            padded_duration_seconds=padded_duration_seconds,
            temporary=True,
        )


DefaultASRAudioPreparer = TorchAudioPreparer


class GraniteSpeechTransformersBackend:
    name = "granite-transformers"

    def __init__(
        self,
        model: str,
        device: str = "auto",
        *,
        local_files_only: bool = True,
        cache_dir: str = "",
    ):
        self.model = model
        self.device = device
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._loaded = None
        self.audio_preparer = TorchAudioPreparer()

    def _device(self):
        return resolve_torch_device(self.device)

    def _load(self, device: str):
        if self._loaded is not None:
            return self._loaded
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers, torch, and torchaudio are required. Install transclip[models].") from exc

        dtype = _granite_transformers_dtype(torch, device)
        processor = AutoProcessor.from_pretrained(
            self.model,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir or None,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model,
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir or None,
        )
        model.to(device)
        model.eval()
        self._loaded = (processor, processor.tokenizer, model)
        return self._loaded

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult:
        timings: dict[str, float] = {}
        device = self._device()
        with timed_ms(timings, "asr"):
            import torch

            processor, tokenizer, model = self._load(device)
            audio = self.audio_preparer.prepare(wav_path)
            audio_seconds = audio.wav.shape[-1] / audio.sample_rate
            max_new_tokens = max(
                AR_MIN_NEW_TOKENS,
                int(audio_seconds * AR_TOKENS_PER_AUDIO_SECOND) + 64,
            )
            prompt = granite_user_prompt(keywords)
            chat = [{"role": "user", "content": f"<|audio|>{prompt}"}]
            templated = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = processor(
                templated,
                audio.wav,
                device=device,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                model_outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
            if new_tokens.shape[-1] >= max_new_tokens:
                logger.warning(
                    "AR generation hit max_new_tokens=%d for %.0fs of audio; transcript may be truncated",
                    max_new_tokens,
                    audio_seconds,
                )
            decoded = tokenizer.batch_decode(
                new_tokens,
                add_special_tokens=False,
                skip_special_tokens=True,
            )
        return TranscriptionResult(decoded[0].strip(), timings, self.name, self.model)


GRANITE_NAR_SAMPLE_RATE = 16000
GRANITE_NAR_BUCKET_SECONDS = 2.0


class GraniteSpeechNarTransformersBackend:
    name = "granite-nar-transformers"

    def __init__(
        self,
        model: str,
        device: str = "auto",
        *,
        local_files_only: bool = True,
        cache_dir: str = "",
    ):
        self.model = model
        self.device = device
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._loaded = None
        self.audio_preparer = TorchAudioPreparer()

    def _device(self):
        return resolve_torch_device(self.device)

    def _load(self, device: str):
        if self._loaded is not None:
            return self._loaded
        try:
            import os

            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers, torch, and torchaudio are required. Install transclip[models].") from exc

        dtype = _granite_nar_dtype(torch, device)
        _configure_rocm_nar_attention_env(os, torch, device)
        model = AutoModel.from_pretrained(
            self.model,
            trust_remote_code=True,
            dtype=dtype,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir or None,
        )
        model.to(device)
        model.eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir or None,
        )
        self._loaded = (feature_extractor, model)
        return self._loaded

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult:
        del keywords
        audio = self.audio_preparer.prepare(wav_path)
        return self.transcribe_waveform(audio.wav.squeeze(0), sample_rate=audio.sample_rate)

    def transcribe_waveform(self, waveform: Any, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe a mono float32 waveform (numpy array or torch tensor); resamples to 16 kHz."""
        timings: dict[str, float] = {}
        device = self._device()
        with timed_ms(timings, "asr"):
            import torch

            feature_extractor, model = self._load(device)
            if not torch.is_tensor(waveform):
                waveform = torch.from_numpy(waveform)
            if sample_rate != GRANITE_NAR_SAMPLE_RATE:
                import torchaudio

                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, GRANITE_NAR_SAMPLE_RATE
                )
                sample_rate = GRANITE_NAR_SAMPLE_RATE
            waveform = _pad_nar_waveform_to_bucket(waveform, sample_rate=sample_rate)
            inputs = feature_extractor([waveform], device=device)
            with torch.inference_mode():
                output = model.generate(**inputs)
        return TranscriptionResult(output.text_preds[0].strip(), timings, self.name, self.model)


class MlxAudioASRBackend:
    name = "mlx-audio"

    def __init__(
        self,
        model: str,
        settings: Settings | None = None,
        *,
        local_files_only: bool = True,
        cache_dir: str = "",
        validate_cache: bool = False,
    ):
        self.model = model
        self.settings = settings
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._resolved_path: str | None = None
        self._loaded_model: Any | None = None
        self._model_lock = threading.RLock()
        self.audio_preparer = PathAudioPreparer(
            bucket_seconds=MLX_AUDIO_BUCKET_SECONDS,
            minimum_seconds=MLX_MIN_AUDIO_SECONDS,
            short_bucket_seconds=MLX_SHORT_AUDIO_BUCKET_SECONDS,
            short_bucket_max_seconds=MLX_SHORT_AUDIO_BUCKET_MAX_SECONDS,
        )
        if validate_cache:
            self._model_path()

    def _model_path(self) -> str:
        if self._resolved_path:
            return self._resolved_path
        settings = self.settings
        if self.local_files_only and settings is not None:
            snapshot = mlx_snapshot_path(self.model, settings)
            if snapshot is not None:
                self._resolved_path = str(snapshot)
                return self._resolved_path
            cache_path = model_cache_path(self.model, settings)
            if cache_path.exists():
                self._resolved_path = str(cache_path)
                return self._resolved_path
            raise RuntimeError(
                f"Local MLX model artifacts missing for {self.model}. "
                f"Run: transclip models prefetch --model {self.model}"
            )
        self._resolved_path = self.model
        return self._resolved_path

    def _load_model(self) -> Any:
        with self._model_lock:
            if self._loaded_model is not None:
                return self._loaded_model
            self._loaded_model = load_mlx_model(self._model_path())
            return self._loaded_model

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult:
        del keywords
        timings: dict[str, float] = {}
        audio: PreparedPathAudio | None = None
        with timed_ms(timings, "asr"):
            with timed_ms(timings, "model_load"):
                model = self._load_model()
            with timed_ms(timings, "audio_prepare"):
                audio = self.audio_preparer.prepare(wav_path)
            try:
                with tempfile.TemporaryDirectory(prefix="transclip-mlx-") as tmp:
                    output_stem = str(Path(tmp) / "transcript")
                    with timed_ms(timings, "generate_write"):
                        result = generate_transcription(
                            model,
                            audio.wav_path,
                            output_stem,
                            language=self.settings.language if self.settings else None,
                            temperature=0.0,
                            return_timestamps=False,
                            condition_on_previous_text=False,
                            sample_len=_mlx_interactive_sample_len(
                                getattr(
                                    audio,
                                    "padded_duration_seconds",
                                    audio.duration_seconds,
                                ),
                                default_sample_len=_mlx_default_sample_len(model),
                            ),
                        )
                    text = getattr(result, "text", None) or str(result)
            finally:
                if audio is not None and getattr(audio, "temporary", False):
                    audio.wav_path.unlink(missing_ok=True)
        return TranscriptionResult(text.strip(), timings, self.name, self.model)


class FileTranscriptASRBackend:
    name = "test-file"

    def __init__(self, transcript_path: Path):
        self.transcript_path = transcript_path
        self.model = f"file:{transcript_path}"

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult:
        del wav_path, keywords
        timings: dict[str, float] = {}
        with timed_ms(timings, "asr"):
            text = self.transcript_path.read_text(encoding="utf-8")
        return TranscriptionResult(text.strip(), timings, self.name, self.model)


def build_asr_backend(
    settings: Settings,
    runtime: PlatformRuntime | None = None,
) -> ASRBackend:
    if settings.asr_backend.startswith("file:"):
        return FileTranscriptASRBackend(Path(settings.asr_backend.removeprefix("file:")))
    backend_kind = validate_asr_model_backend(settings.asr_backend, settings.asr_model, runtime)
    entry = resolve_catalog_entry(settings, runtime)
    if entry is None:
        raise ValueError(f"Unsupported ASR configuration: {settings.asr_backend} / {settings.asr_model}")

    torch_device = "auto" if backend_kind == "granite" and settings.asr_device == "mlx" else settings.asr_device
    cache_options = {
        "local_files_only": settings.models_local_files_only,
        "cache_dir": settings.model_cache_dir,
    }
    if backend_kind == "granite_nar":
        backend = GraniteSpeechNarTransformersBackend(settings.asr_model, torch_device, **cache_options)
    elif backend_kind in {"mlx_audio_whisper", "granite_mlx", "granite_nar_mlx"}:
        backend = MlxAudioASRBackend(
            settings.asr_model,
            settings,
            **cache_options,
            validate_cache=settings.models_local_files_only,
        )
    else:
        backend = GraniteSpeechTransformersBackend(settings.asr_model, torch_device, **cache_options)
    return backend


def _pad_nar_waveform_to_bucket(
    waveform: Any,
    sample_rate: int,
    bucket_seconds: float = GRANITE_NAR_BUCKET_SECONDS,
) -> Any:
    """Pad NAR inputs to stable tensor buckets to avoid first-use shape compiles."""
    bucket_samples = max(1, int(sample_rate * bucket_seconds))
    length = int(waveform.shape[-1] if hasattr(waveform, "shape") else len(waveform))
    if length == 0 or length % bucket_samples == 0:
        return waveform
    target = math.ceil(length / bucket_samples) * bucket_samples
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(waveform):
        padded = waveform.new_zeros(target)
        padded[:length] = waveform
        return padded

    import numpy as np

    padded = np.zeros(target, dtype=getattr(waveform, "dtype", np.float32))
    padded[:length] = waveform
    return padded


def granite_user_prompt(keywords: list[str] | None = None) -> str:
    if keywords:
        keyword_text = ", ".join(keyword.strip() for keyword in keywords if keyword.strip())
        if keyword_text:
            return f"transcribe the speech to text. Keywords: {keyword_text}"
    return "transcribe the speech with proper punctuation and capitalization."


def _granite_transformers_dtype(torch, device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps" and _mps_bfloat16_supported():
        return torch.bfloat16
    return torch.float32


def _mps_bfloat16_supported() -> bool:
    version = py_platform.mac_ver()[0]
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return True
    return major >= 14


def _granite_nar_dtype(torch, device: str):
    if device != "cuda":
        return torch.float32
    if getattr(torch.version, "hip", None):
        return torch.float32
    return torch.bfloat16


def _configure_rocm_nar_attention_env(os_module, torch, device: str) -> None:
    if device == "cuda" and getattr(torch.version, "hip", None):
        os_module.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
        os_module.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def _linear_resample(samples: Any, source_rate: int, target_rate: int) -> Any:
    if source_rate == target_rate:
        return samples
    import numpy as np

    if len(samples) == 0:
        return samples
    target_length = max(1, round(len(samples) * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, num=len(samples), endpoint=True)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(target_positions, source_positions, samples).astype(samples.dtype, copy=False)


def _trim_edge_silence(
    samples: Any,
    sample_rate: int,
    *,
    threshold: float = EDGE_SILENCE_TRIM_THRESHOLD,
    padding_seconds: float = EDGE_SILENCE_TRIM_PADDING_SECONDS,
) -> tuple[Any, bool]:
    import numpy as np

    if len(samples) == 0:
        return samples, False
    active = np.flatnonzero(np.abs(samples) > threshold)
    if active.size == 0:
        return samples, False
    padding = max(0, round(sample_rate * padding_seconds))
    start = max(0, int(active[0]) - padding)
    end = min(len(samples), int(active[-1]) + padding + 1)
    if start == 0 and end == len(samples):
        return samples, False
    return samples[start:end], True


def _pad_to_audio_bucket(
    samples: Any,
    sample_rate: int,
    *,
    bucket_seconds: float,
    minimum_seconds: float,
    short_bucket_seconds: float = 0.0,
    short_bucket_max_seconds: float = 0.0,
) -> tuple[Any, bool]:
    if bucket_seconds <= 0 and minimum_seconds <= 0 and short_bucket_seconds <= 0:
        return samples, False
    import numpy as np

    current = len(samples)
    minimum = max(0, round(sample_rate * minimum_seconds))
    target = _audio_bucket_target_samples(
        current,
        sample_rate=sample_rate,
        bucket_seconds=bucket_seconds,
        minimum=minimum,
        short_bucket_seconds=short_bucket_seconds,
        short_bucket_max_seconds=short_bucket_max_seconds,
    )
    if target <= current:
        return samples, False
    padded = np.zeros(target, dtype=getattr(samples, "dtype", np.float32))
    padded[:current] = samples
    return padded, True


def _audio_bucket_target_samples(
    current: int,
    *,
    sample_rate: int,
    bucket_seconds: float,
    minimum: int,
    short_bucket_seconds: float,
    short_bucket_max_seconds: float,
) -> int:
    if current <= minimum and minimum > 0:
        return minimum
    if short_bucket_seconds > 0 and short_bucket_max_seconds > 0:
        short_bucket = max(1, round(sample_rate * short_bucket_seconds))
        short_maximum = max(1, round(sample_rate * short_bucket_max_seconds))
        if current <= short_maximum:
            return max(minimum, math.ceil(max(current, 1) / short_bucket) * short_bucket)
    if bucket_seconds > 0:
        bucket = max(1, round(sample_rate * bucket_seconds))
        return max(minimum, math.ceil(max(current, 1) / bucket) * bucket)
    return minimum


def _mlx_default_sample_len(model: Any) -> int:
    dims = getattr(model, "dims", None)
    n_text_ctx = getattr(dims, "n_text_ctx", None)
    if isinstance(n_text_ctx, int) and n_text_ctx > 0:
        return max(1, n_text_ctx // 2)
    return 224


def _mlx_interactive_sample_len(audio_seconds: float, *, default_sample_len: int) -> int:
    scaled = math.ceil(max(0.0, audio_seconds) * MLX_TOKENS_PER_AUDIO_SECOND)
    desired = max(MLX_MIN_SAMPLE_LEN, scaled + MLX_SAMPLE_LEN_PADDING_TOKENS)
    return min(max(1, default_sample_len), desired)
