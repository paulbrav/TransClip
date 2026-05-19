import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tests.platform_helpers import linux_runtime
from transclip.asr import (
    FileTranscriptASRBackend,
    GraniteSpeechNarTransformersBackend,
    GraniteSpeechTransformersBackend,
    MlxAudioASRBackend,
    PathAudioPreparer,
    TorchAudioPreparer,
    _configure_rocm_nar_attention_env,
    _granite_nar_dtype,
    _granite_transformers_dtype,
    build_asr_backend,
    granite_user_prompt,
)
from transclip.settings import Settings


class ASRTests(unittest.TestCase):
    linux = linux_runtime()

    def test_granite_prompt_requests_punctuation(self):
        self.assertEqual(
            granite_user_prompt(),
            "transcribe the speech with proper punctuation and capitalization.",
        )

    def test_granite_prompt_uses_keyword_biasing_format(self):
        self.assertEqual(
            granite_user_prompt(["PyTorch", "ROCm", "", " gfx1151 "]),
            "transcribe the speech to text. Keywords: PyTorch, ROCm, gfx1151",
        )

    def test_backend_selection(self):
        backend = build_asr_backend(Settings(model_cache_dir="/models"), runtime=self.linux)
        self.assertIsInstance(backend, GraniteSpeechNarTransformersBackend)
        self.assertTrue(backend.local_files_only)
        self.assertEqual(backend.cache_dir, "/models")
        ar_backend = build_asr_backend(
            Settings(
                asr_backend="granite",
                asr_model="ibm-granite/granite-speech-4.1-2b",
            ),
            runtime=self.linux,
        )
        self.assertIsInstance(ar_backend, GraniteSpeechTransformersBackend)
        self.assertIsInstance(
            build_asr_backend(Settings(asr_backend="file:/tmp/transcript.txt"), runtime=self.linux),
            FileTranscriptASRBackend,
        )
        nar_backend = build_asr_backend(
            Settings(
                asr_backend="granite_nar",
                asr_model="ibm-granite/granite-speech-4.1-2b-nar",
                model_cache_dir="/models",
            ),
            runtime=self.linux,
        )
        self.assertIsInstance(nar_backend, GraniteSpeechNarTransformersBackend)
        self.assertTrue(nar_backend.local_files_only)
        self.assertEqual(nar_backend.cache_dir, "/models")

    def test_non_granite_model_is_rejected(self):
        with self.assertRaises(ValueError):
            build_asr_backend(Settings(asr_model="openai/whisper-tiny"), runtime=self.linux)
        with self.assertRaises(ValueError):
            build_asr_backend(
                Settings(
                    asr_backend="granite_nar",
                    asr_model="ibm-granite/granite-speech-4.1-2b",
                ),
                runtime=self.linux,
            )
        with self.assertRaises(ValueError):
            build_asr_backend(
                Settings(
                    asr_backend="granite",
                    asr_model="ibm-granite/granite-speech-4.1-2b-nar",
                ),
                runtime=self.linux,
            )

    def test_granite_nar_uses_float32_on_rocm(self):
        torch = SimpleNamespace(
            bfloat16="bfloat16",
            float32="float32",
            version=SimpleNamespace(hip="6.4"),
        )

        self.assertEqual(_granite_nar_dtype(torch, "cuda"), "float32")
        self.assertEqual(_granite_nar_dtype(torch, "cpu"), "float32")

        torch.version.hip = None
        self.assertEqual(_granite_nar_dtype(torch, "cuda"), "bfloat16")

    def test_granite_transformers_uses_bfloat16_on_mps(self):
        torch = SimpleNamespace(
            bfloat16="bfloat16",
            float32="float32",
        )

        self.assertEqual(_granite_transformers_dtype(torch, "mps"), "bfloat16")
        self.assertEqual(_granite_transformers_dtype(torch, "cpu"), "float32")

    def test_granite_nar_sets_rocm_attention_environment(self):
        environ = {}
        os_module = SimpleNamespace(environ=environ)
        torch = SimpleNamespace(version=SimpleNamespace(hip="6.4"))

        _configure_rocm_nar_attention_env(os_module, torch, "cuda")

        self.assertEqual(environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"], "TRUE")
        self.assertEqual(environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"], "1")

        environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"
        _configure_rocm_nar_attention_env(os_module, torch, "cuda")
        self.assertEqual(environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"], "0")

    def test_granite_nar_leaves_non_rocm_environment_unchanged(self):
        environ = {}
        os_module = SimpleNamespace(environ=environ)
        torch = SimpleNamespace(version=SimpleNamespace(hip=None))

        _configure_rocm_nar_attention_env(os_module, torch, "cuda")
        _configure_rocm_nar_attention_env(os_module, torch, "cpu")

        self.assertEqual(environ, {})

    def test_audio_preparer_folds_channels_and_resamples_without_model_runtime(self):
        samples = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        resample_calls = []

        class FakeTensor:
            def __init__(self, data):
                self.data = np.array(data)

            @property
            def shape(self):
                return self.data.shape

            def mean(self, dim, keepdim):
                return FakeTensor(self.data.mean(axis=dim, keepdims=keepdim))

            def squeeze(self, dim):
                return FakeTensor(np.squeeze(self.data, axis=dim))

        fake_soundfile = SimpleNamespace(
            read=lambda *_args, **_kwargs: (samples, 8000),
        )
        fake_torch = SimpleNamespace(
            from_numpy=lambda value: FakeTensor(value),
        )

        def resample(wav, source_rate, target_rate):
            resample_calls.append((source_rate, target_rate))
            return wav

        fake_torchaudio = SimpleNamespace(functional=SimpleNamespace(resample=resample))

        with patch.dict(
            "sys.modules",
            {
                "soundfile": fake_soundfile,
                "torch": fake_torch,
                "torchaudio": fake_torchaudio,
            },
        ):
            audio = TorchAudioPreparer().prepare(Path("sample.wav"))

        self.assertEqual(audio.sample_rate, 16000)
        np.testing.assert_allclose(audio.wav.data, np.array([[2.0, 6.0]], dtype=np.float32))
        self.assertEqual(resample_calls, [(8000, 16000)])

    def test_path_preparer_resamples_non_target_rate(self):
        samples = np.linspace(0.0, 1.0, num=480, dtype=np.float32)
        captured: dict[str, object] = {}

        def write(_path, data, sample_rate):
            captured["data"] = data
            captured["sample_rate"] = sample_rate

        fake_soundfile = SimpleNamespace(
            read=lambda *_args, **_kwargs: (samples[:, None], 48000),
            write=write,
        )

        with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
            prepared = PathAudioPreparer().prepare(Path("sample.wav"))

        self.assertEqual(prepared.sample_rate, 16000)
        self.assertNotEqual(prepared.wav_path, Path("sample.wav"))
        self.assertEqual(captured["sample_rate"], 16000)
        self.assertEqual(len(captured["data"]), 160)
        self.assertTrue(prepared.temporary)

    def test_path_preparer_returns_existing_mono_wav(self):
        wav_path = Path("/tmp/sample.wav")
        samples = np.array([[0.1], [0.2]], dtype=np.float32)
        fake_soundfile = SimpleNamespace(read=lambda *_args, **_kwargs: (samples, 16000))
        with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
            prepared = PathAudioPreparer().prepare(wav_path)
        self.assertEqual(prepared.wav_path, wav_path)
        self.assertEqual(prepared.sample_rate, 16000)
        self.assertFalse(prepared.temporary)

    def test_mlx_audio_backend_uses_current_generate_api(self):
        calls = []
        fake_generate = SimpleNamespace(
            generate_transcription=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(text="hello"),
        )
        backend = MlxAudioASRBackend("mlx/model", "mlx_audio_whisper")
        backend.local_files_only = False
        backend.audio_preparer = SimpleNamespace(prepare=lambda _path: SimpleNamespace(wav_path=Path("prepared.wav")))

        with patch.dict("sys.modules", {"mlx_audio.stt.generate": fake_generate}):
            result = backend.transcribe(Path("input.wav"))

        self.assertEqual(result.text, "hello")
        self.assertEqual(calls, [{"model": "mlx/model", "audio": "prepared.wav"}])

    def test_mlx_audio_backend_removes_temporary_prepared_wav(self):
        temp_path = Path("/tmp/transclip-test-prepared.wav")
        temp_path.write_bytes(b"wav")
        fake_generate = SimpleNamespace(
            generate_transcription=lambda **_kwargs: SimpleNamespace(text="hello"),
        )
        backend = MlxAudioASRBackend("mlx/model", "mlx_audio_whisper")
        backend.local_files_only = False
        backend.audio_preparer = SimpleNamespace(
            prepare=lambda _path: SimpleNamespace(wav_path=temp_path, temporary=True),
        )

        try:
            with patch.dict("sys.modules", {"mlx_audio.stt.generate": fake_generate}):
                result = backend.transcribe(Path("input.wav"))
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertEqual(result.text, "hello")
        self.assertFalse(temp_path.exists())

    def test_granite_nar_transcribe_uses_processor_and_model_transcribe(self):
        backend = GraniteSpeechNarTransformersBackend("ibm-granite/granite-speech-4.1-2b-nar", "cpu")
        waveform = SimpleNamespace()
        class Processor:
            def __call__(self, waveforms, device):
                return {"waveforms": waveforms, "device": device}

            def batch_decode(self, preds):
                return [f"decoded:{preds[0]}"]

        processor = Processor()
        model = SimpleNamespace(
            transcribe=lambda **_kwargs: SimpleNamespace(preds=["pred-text"]),
        )
        backend._loaded = (processor, model)
        backend.audio_preparer = SimpleNamespace(
            prepare=lambda _path: SimpleNamespace(wav=SimpleNamespace(squeeze=lambda _dim: waveform)),
        )

        with patch.object(backend, "_device", return_value="cpu"):
            result = backend.transcribe(Path("sample.wav"))

        self.assertEqual(result.text, "decoded:pred-text")

    def test_granite_nar_load_uses_auto_processor_and_flash_attention(self):
        backend = GraniteSpeechNarTransformersBackend("ibm-granite/granite-speech-4.1-2b-nar", "cuda")
        captured: dict[str, object] = {}

        class FakeModel:
            def eval(self):
                return self

        def from_pretrained(_model, **kwargs):
            captured["model_kwargs"] = kwargs
            return FakeModel()

        def processor_from_pretrained(_model, **kwargs):
            captured["processor_kwargs"] = kwargs
            return "processor"

        fake_transformers = SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=from_pretrained),
            AutoProcessor=SimpleNamespace(from_pretrained=processor_from_pretrained),
        )
        fake_torch = SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            version=SimpleNamespace(hip=None),
        )

        with (
            patch.dict("sys.modules", {"transformers": fake_transformers, "torch": fake_torch}),
            patch("transclip.asr._configure_rocm_nar_attention_env"),
            patch("transclip.asr._granite_nar_dtype", return_value="bf16"),
        ):
            processor, model = backend._load("cuda")

        self.assertEqual(processor, "processor")
        self.assertIsInstance(model, FakeModel)
        self.assertEqual(captured["model_kwargs"]["attn_implementation"], "flash_attention_2")
        self.assertEqual(captured["model_kwargs"]["device_map"], "cuda")

    def test_granite_nar_load_does_not_require_flash_attention_on_cpu(self):
        backend = GraniteSpeechNarTransformersBackend("ibm-granite/granite-speech-4.1-2b-nar", "cpu")
        captured: dict[str, object] = {}

        class FakeModel:
            def to(self, device):
                captured["to_device"] = device
                return self

            def eval(self):
                return self

        def from_pretrained(_model, **kwargs):
            captured["model_kwargs"] = kwargs
            return FakeModel()

        fake_transformers = SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=from_pretrained),
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: "processor"),
        )
        fake_torch = SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            version=SimpleNamespace(hip=None),
        )

        with (
            patch.dict("sys.modules", {"transformers": fake_transformers, "torch": fake_torch}),
            patch("transclip.asr._configure_rocm_nar_attention_env"),
            patch("transclip.asr._granite_nar_dtype", return_value="fp32"),
        ):
            backend._load("cpu")

        self.assertNotIn("attn_implementation", captured["model_kwargs"])
        self.assertNotIn("device_map", captured["model_kwargs"])
        self.assertEqual(captured["to_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
