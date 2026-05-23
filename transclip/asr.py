from __future__ import annotations

import platform as py_platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from transclip.platform.runtime import PlatformRuntime

from .device import resolve_torch_device
from .mlx_audio_compat import generate_transcription
from .models import (
    mlx_snapshot_path,
    model_cache_path,
    resolve_catalog_entry,
    validate_asr_model_backend,
)
from .settings import Settings
from .timing import timed_ms


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
    temporary: bool = False


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
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.loader = AudioLoader(target_sample_rate)

    def prepare(self, wav_path: Path) -> PreparedPathAudio:
        samples, sample_rate = self.loader.load_samples(wav_path)
        if sample_rate == self.target_sample_rate and samples.shape[1] == 1:
            return PreparedPathAudio(wav_path=wav_path, sample_rate=sample_rate)

        import soundfile as sf

        mono = self.loader.fold_mono(samples)
        if sample_rate != self.target_sample_rate:
            mono = _linear_resample(mono, sample_rate, self.target_sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = Path(handle.name)
        sf.write(str(output), mono, self.target_sample_rate)
        return PreparedPathAudio(wav_path=output, sample_rate=self.target_sample_rate, temporary=True)


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
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1,
                )
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
            decoded = tokenizer.batch_decode(
                new_tokens,
                add_special_tokens=False,
                skip_special_tokens=True,
            )
        return TranscriptionResult(decoded[0].strip(), timings, self.name, self.model)


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
        timings: dict[str, float] = {}
        device = self._device()
        with timed_ms(timings, "asr"):
            import torch

            feature_extractor, model = self._load(device)
            audio = self.audio_preparer.prepare(wav_path)
            waveform = audio.wav.squeeze(0)
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
        self.audio_preparer = PathAudioPreparer()
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

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult:
        del keywords
        timings: dict[str, float] = {}
        with timed_ms(timings, "asr"):
            model_path = self._model_path()
            audio = self.audio_preparer.prepare(wav_path)
            try:
                with tempfile.TemporaryDirectory(prefix="transclip-mlx-") as tmp:
                    output_stem = str(Path(tmp) / "transcript")
                    result = generate_transcription(model_path, audio.wav_path, output_stem)
                    text = getattr(result, "text", None) or str(result)
            finally:
                if getattr(audio, "temporary", False):
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
    elif backend_kind in {"mlx_audio_whisper", "granite_mlx"}:
        backend = MlxAudioASRBackend(
            settings.asr_model,
            settings,
            **cache_options,
            validate_cache=settings.models_local_files_only,
        )
    else:
        backend = GraniteSpeechTransformersBackend(settings.asr_model, torch_device, **cache_options)
    return backend


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
