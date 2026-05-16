from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .device import resolve_torch_device
from .glossary import keyword_prompt
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

    def transcribe(self, wav_path: Path, keywords: list[str]) -> TranscriptionResult:
        ...


class GraniteSpeechTransformersBackend:
    name = "granite-transformers"

    def __init__(self, model: str, device: str = "auto"):
        self.model = model
        self.device = device
        self.local_files_only = True
        self.cache_dir = ""
        self._loaded = None

    def _device(self):
        return resolve_torch_device(self.device)

    def _load(self, device: str):
        if self._loaded is not None:
            return self._loaded
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers, torch, and torchaudio are required. Install granite-speach[models]."
            ) from exc

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

    def transcribe(self, wav_path: Path, keywords: list[str]) -> TranscriptionResult:
        timings: dict[str, float] = {}
        device = self._device()
        with timed_ms(timings, "asr"):
            import torch
            import soundfile as sf

            processor, tokenizer, model = self._load(device)
            samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
            wav = torch.from_numpy(samples.T)
            if wav.shape[0] != 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                import torchaudio

                wav = torchaudio.functional.resample(wav, sample_rate, 16000)
            prompt = granite_user_prompt(keywords)
            chat = [{"role": "user", "content": f"<|audio|>{prompt}"}]
            templated = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = processor(
                templated,
                wav,
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
            raise RuntimeError(
                "transformers, torch, and torchaudio are required. Install granite-speach[models]."
            ) from exc

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if device == "cuda" and getattr(torch.version, "hip", None):
            os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
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

    def transcribe(self, wav_path: Path, keywords: list[str]) -> TranscriptionResult:
        del keywords
        timings: dict[str, float] = {}
        device = self._device()
        with timed_ms(timings, "asr"):
            import soundfile as sf
            import torch

            feature_extractor, model = self._load(device)
            samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
            wav = torch.from_numpy(samples.T)
            if wav.shape[0] != 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                import torchaudio

                wav = torchaudio.functional.resample(wav, sample_rate, 16000)
            waveform = wav.squeeze(0)
            inputs = feature_extractor([waveform], device=device)
            with torch.inference_mode():
                output = model.generate(**inputs)
        return TranscriptionResult(output.text_preds[0].strip(), timings, self.name, self.model)


class FileTranscriptASRBackend:
    name = "test-file"

    def __init__(self, transcript_path: Path):
        self.transcript_path = transcript_path
        self.model = f"file:{transcript_path}"

    def transcribe(self, wav_path: Path, keywords: list[str]) -> TranscriptionResult:
        del wav_path, keywords
        timings: dict[str, float] = {}
        with timed_ms(timings, "asr"):
            text = self.transcript_path.read_text(encoding="utf-8")
        return TranscriptionResult(text.strip(), timings, self.name, self.model)


def build_asr_backend(settings: Settings) -> ASRBackend:
    if settings.asr_backend.startswith("file:"):
        return FileTranscriptASRBackend(Path(settings.asr_backend.removeprefix("file:")))
    if settings.asr_backend in {"granite_nar", "granite-nar", "nar"}:
        if "granite-speech" not in settings.asr_model or "-nar" not in settings.asr_model:
            raise ValueError("Granite NAR ASR requires an ibm-granite granite-speech NAR model")
        backend = GraniteSpeechNarTransformersBackend(settings.asr_model, settings.asr_device)
        backend.local_files_only = settings.models_local_files_only
        backend.cache_dir = settings.model_cache_dir
        return backend
    if settings.asr_backend not in {"granite", "transformers"}:
        raise ValueError(f"Unsupported ASR backend: {settings.asr_backend}")
    if "-nar" in settings.asr_model:
        raise ValueError('Use asr_backend = "granite_nar" with Granite NAR models')
    if "granite-speech" not in settings.asr_model:
        raise ValueError("V1 ASR requires an ibm-granite granite-speech model")
    backend = GraniteSpeechTransformersBackend(settings.asr_model, settings.asr_device)
    backend.local_files_only = settings.models_local_files_only
    backend.cache_dir = settings.model_cache_dir
    return backend


def granite_user_prompt(keywords: list[str]) -> str:
    return keyword_prompt(keywords)
