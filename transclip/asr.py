from __future__ import annotations

import logging
import math
import platform as py_platform
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

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
from .openvino_device import download_openvino_snapshot, resolve_openvino_device
from .settings import Settings
from .timing import timed_ms

if TYPE_CHECKING:
    from .device import TorchDevice

logger = logging.getLogger(__name__)

# The backend identity string reported by an ASR backend (ASRBackend.name) and
# carried on TranscriptionResult.backend / the /transcribe and /health wire
# fields. Distinct from the catalog/normalized ASRBackendKind (e.g. "granite").
ASRBackendName = Literal[
    "granite-transformers",
    "granite-nar-transformers",
    "mlx-audio",
    "openvino-whisper",
    "test-file",
]

AR_TOKENS_PER_AUDIO_SECOND = 10
AR_MIN_NEW_TOKENS = 200


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    timings_ms: dict[str, float]
    # The backend identity reported on the wire. ASRBackend implementations pass
    # their ASRBackendName; the incremental path passes its own free-form label,
    # so this stays a plain str.
    backend: str
    model: str


class ASRBackend(Protocol):
    name: ASRBackendName
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
    name: ASRBackendName = "granite-transformers"

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

    def _load(self, device: TorchDevice):
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
    name: ASRBackendName = "granite-nar-transformers"

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

    def _load(self, device: TorchDevice):
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
    name: ASRBackendName = "mlx-audio"

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
                    with timed_ms(timings, "generate"):
                        result = generate_transcription(
                            model,
                            audio.wav_path,
                            output_stem,
                            language=self.settings.language if self.settings else None,
                        )
                    result_text = getattr(result, "text", None)
                    text = str(result) if result_text is None else str(result_text)
            finally:
                if audio is not None and getattr(audio, "temporary", False):
                    audio.wav_path.unlink(missing_ok=True)
        return TranscriptionResult(text.strip(), timings, self.name, self.model)


class OpenVINOWhisperBackend:
    """Whisper ASR accelerated on Intel CPU/iGPU/NPU via OpenVINO GenAI.

    Mirrors MlxAudioASRBackend: lazy, lock-guarded pipeline load from a local HF
    snapshot. OpenVINO uses its own device namespace (CPU/GPU/NPU/AUTO) resolved
    by resolve_openvino_device, kept separate from the torch device path.
    """

    name: ASRBackendName = "openvino-whisper"

    def __init__(
        self,
        model: str,
        settings: Settings | None = None,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        cache_dir: str = "",
        validate_cache: bool = False,
    ):
        self.model = model
        self.settings = settings
        self.device = device
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._resolved_path: str | None = None
        self._resolved_device: str | None = None
        self._pipeline: Any | None = None
        self._model_lock = threading.RLock()
        self.audio_preparer = PathAudioPreparer()
        self._keywords_warned = False
        if validate_cache:
            self._model_path()

    def _model_path(self) -> str:
        if self._resolved_path:
            return self._resolved_path
        settings = self.settings
        if settings is not None:
            snapshot = mlx_snapshot_path(self.model, settings)
            if snapshot is not None:
                self._resolved_path = str(snapshot)
                return self._resolved_path
        if self.local_files_only:
            raise RuntimeError(
                f"Local OpenVINO model artifacts missing for {self.model}. "
                f"Run: transclip models prefetch --model {self.model}"
            )
        self._resolved_path = self._download_snapshot()
        return self._resolved_path

    def _download_snapshot(self) -> str:
        return download_openvino_snapshot(self.model, self.cache_dir)

    def _ov_device(self) -> str:
        if self._resolved_device is None:
            self._resolved_device = resolve_openvino_device(self.device)
        return self._resolved_device

    def _load(self) -> Any:
        with self._model_lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import openvino_genai as ov_genai
            except ImportError as exc:
                raise RuntimeError(
                    "openvino-genai is required for the OpenVINO ASR backend. Install transclip[openvino]."
                ) from exc
            device = self._ov_device()
            config: dict[str, Any] = {}
            if device == "NPU":
                # The NPU plugin requires static shapes for the Whisper pipeline.
                config["STATIC_PIPELINE"] = True
            self._pipeline = ov_genai.WhisperPipeline(self._model_path(), device, **config)
            return self._pipeline

    def _read_audio(self, wav_path: Path) -> Any:
        import numpy as np

        loader = self.audio_preparer.loader
        samples, sample_rate = loader.load_samples(wav_path)
        mono = loader.fold_mono(samples)
        if sample_rate != loader.target_sample_rate:
            mono = _linear_resample(mono, sample_rate, loader.target_sample_rate)
        return np.ascontiguousarray(mono, dtype=np.float32)

    def _generate(self, pipeline: Any, samples: Any) -> Any:
        kwargs: dict[str, Any] = {}
        language = self.settings.language if self.settings else None
        token = _whisper_language_token(language) if language else ""
        if token:
            kwargs["language"] = token
            kwargs["task"] = "transcribe"
        return pipeline.generate(samples, **kwargs)

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult:
        if keywords and not self._keywords_warned:
            logger.debug("OpenVINO Whisper does not support keyword biasing; keywords are ignored")
            self._keywords_warned = True
        del keywords
        timings: dict[str, float] = {}
        with timed_ms(timings, "asr"):
            with timed_ms(timings, "model_load"):
                pipeline = self._load()
            with timed_ms(timings, "audio_prepare"):
                samples = self._read_audio(wav_path)
            with timed_ms(timings, "generate"):
                result = self._generate(pipeline, samples)
        return TranscriptionResult(_openvino_result_text(result).strip(), timings, self.name, self.model)


class FileTranscriptASRBackend:
    name: ASRBackendName = "test-file"

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
    elif backend_kind == "openvino_whisper":
        backend = OpenVINOWhisperBackend(
            settings.asr_model,
            settings,
            device=settings.asr_device,
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


def _whisper_language_token(language: str | None) -> str:
    lang = (language or "").strip()
    if not lang or lang.startswith("<|"):
        return lang
    return f"<|{lang}|>"


def _openvino_result_text(result: Any) -> str:
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0])
    return str(result)


def granite_user_prompt(keywords: list[str] | None = None) -> str:
    if keywords:
        keyword_text = ", ".join(keyword.strip() for keyword in keywords if keyword.strip())
        if keyword_text:
            return f"transcribe the speech to text. Keywords: {keyword_text}"
    return "transcribe the speech with proper punctuation and capitalization."


def _granite_transformers_dtype(torch, device: TorchDevice):
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


def _granite_nar_dtype(torch, device: TorchDevice):
    if device != "cuda":
        return torch.float32
    if getattr(torch.version, "hip", None):
        return torch.float32
    return torch.bfloat16


def _configure_rocm_nar_attention_env(os_module, torch, device: TorchDevice) -> None:
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
