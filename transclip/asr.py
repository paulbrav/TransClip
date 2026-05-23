from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .device import resolve_torch_device
from .models import validate_asr_model_backend
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


class DefaultASRAudioPreparer:
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def prepare(self, wav_path: Path) -> PreparedAudio:
        import soundfile as sf
        import torch

        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(samples.T)
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sample_rate, self.target_sample_rate)
        return PreparedAudio(wav=wav, sample_rate=self.target_sample_rate)


class GraniteSpeechTransformersBackend:
    name = "granite-transformers"

    def __init__(self, model: str, device: str = "auto"):
        self.model = model
        self.device = device
        self.local_files_only = True
        self.cache_dir = ""
        self._loaded = None
        self.audio_preparer = DefaultASRAudioPreparer()

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

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
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

    def __init__(self, model: str, device: str = "auto"):
        self.model = model
        self.device = device
        self.local_files_only = True
        self.cache_dir = ""
        self._loaded = None
        self.audio_preparer = DefaultASRAudioPreparer()

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


def build_asr_backend(settings: Settings) -> ASRBackend:
    if settings.asr_backend.startswith("file:"):
        return FileTranscriptASRBackend(Path(settings.asr_backend.removeprefix("file:")))
    backend_kind = validate_asr_model_backend(settings.asr_backend, settings.asr_model)
    if backend_kind == "granite_nar":
        backend = GraniteSpeechNarTransformersBackend(settings.asr_model, settings.asr_device)
        backend.local_files_only = settings.models_local_files_only
        backend.cache_dir = settings.model_cache_dir
        return backend
    backend = GraniteSpeechTransformersBackend(settings.asr_model, settings.asr_device)
    backend.local_files_only = settings.models_local_files_only
    backend.cache_dir = settings.model_cache_dir
    return backend


def granite_user_prompt(keywords: list[str] | None = None) -> str:
    if keywords:
        keyword_text = ", ".join(keyword.strip() for keyword in keywords if keyword.strip())
        if keyword_text:
            return f"transcribe the speech to text. Keywords: {keyword_text}"
    return "transcribe the speech with proper punctuation and capitalization."


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
