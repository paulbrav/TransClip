import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from transclip.asr import (
    DefaultASRAudioPreparer,
    FileTranscriptASRBackend,
    GraniteSpeechNarTransformersBackend,
    GraniteSpeechTransformersBackend,
    MlxAudioASRBackend,
    OpenVINOWhisperBackend,
    _configure_rocm_nar_attention_env,
    _granite_nar_dtype,
    _pad_nar_waveform_to_bucket,
    _whisper_language_token,
    build_asr_backend,
    granite_user_prompt,
)
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime


class ASRTests(unittest.TestCase):
    @staticmethod
    def _linux_runtime() -> FakeRuntime:
        return FakeRuntime(system="Linux", home=Path("/home/user"))
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
        runtime = self._linux_runtime()
        backend = build_asr_backend(Settings(model_cache_dir="/models"), runtime=runtime)
        self.assertIsInstance(backend, GraniteSpeechNarTransformersBackend)
        self.assertTrue(backend.local_files_only)
        self.assertEqual(backend.cache_dir, "/models")
        ar_backend = build_asr_backend(
            Settings(
                asr_backend="granite",
                asr_model="ibm-granite/granite-speech-4.1-2b",
            ),
            runtime=runtime,
        )
        self.assertIsInstance(ar_backend, GraniteSpeechTransformersBackend)
        self.assertIsInstance(
            build_asr_backend(Settings(asr_backend="file:/tmp/transcript.txt"), runtime=runtime),
            FileTranscriptASRBackend,
        )
        nar_backend = build_asr_backend(
            Settings(
                asr_backend="granite_nar",
                asr_model="ibm-granite/granite-speech-4.1-2b-nar",
                model_cache_dir="/models",
            ),
            runtime=runtime,
        )
        self.assertIsInstance(nar_backend, GraniteSpeechNarTransformersBackend)
        self.assertTrue(nar_backend.local_files_only)
        self.assertEqual(nar_backend.cache_dir, "/models")

    def test_darwin_arm_selects_mlx_backend(self):
        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="arm64")
        backend = build_asr_backend(
            Settings(
                asr_backend="mlx_audio_whisper",
                asr_model="mlx-community/whisper-large-v3-turbo-asr-fp16",
                models_local_files_only=False,
            ),
            runtime=runtime,
        )
        self.assertIsInstance(backend, MlxAudioASRBackend)

    @staticmethod
    def _windows_runtime() -> FakeRuntime:
        return FakeRuntime(system="Windows", home=Path("C:/Users/test"))

    def test_windows_openvino_backend_selection(self):
        runtime = self._windows_runtime()
        with patch("transclip.device.torch_cuda_usable", return_value=True):
            backend = build_asr_backend(
                Settings(
                    asr_backend="openvino_whisper",
                    asr_model="OpenVINO/whisper-large-v3-int4-ov",
                    asr_device="auto",
                    models_local_files_only=False,
                ),
                runtime=runtime,
            )
        self.assertIsInstance(backend, OpenVINOWhisperBackend)
        self.assertEqual(backend.device, "auto")

    def test_openvino_requires_whisper_model(self):
        runtime = self._windows_runtime()
        with (
            patch("transclip.device.torch_cuda_usable", return_value=True),
            self.assertRaises(ValueError),
        ):
            build_asr_backend(
                Settings(
                    asr_backend="openvino_whisper",
                    asr_model="ibm-granite/granite-speech-4.1-2b",
                ),
                runtime=runtime,
            )

    def test_openvino_transcribe_ignores_keywords_and_uses_pipeline(self):
        captured = {}

        class FakePipeline:
            def generate(self, samples, **kwargs):
                captured["samples_len"] = len(samples)
                captured["kwargs"] = kwargs
                return SimpleNamespace(texts=["  hello world  "])

        backend = OpenVINOWhisperBackend(
            "OpenVINO/whisper-large-v3-int4-ov",
            Settings(language="en"),
            device="openvino:CPU",
            local_files_only=False,
        )
        backend._pipeline = FakePipeline()
        backend._read_audio = lambda _path: np.zeros(16000, dtype=np.float32)

        result = backend.transcribe(Path("a.wav"), keywords=["PyTorch"])

        self.assertEqual(result.text, "hello world")
        self.assertEqual(captured["samples_len"], 16000)
        self.assertEqual(captured["kwargs"].get("language"), "<|en|>")
        self.assertEqual(captured["kwargs"].get("task"), "transcribe")
        self.assertIn("model_load", result.timings_ms)
        self.assertIn("generate", result.timings_ms)

    def test_openvino_load_sets_static_pipeline_on_npu(self):
        captured = {}

        class FakeWhisperPipeline:
            def __init__(self, model_path, device, **kwargs):
                captured["model_path"] = model_path
                captured["device"] = device
                captured["kwargs"] = kwargs

        fake_genai = SimpleNamespace(WhisperPipeline=FakeWhisperPipeline)
        backend = OpenVINOWhisperBackend(
            "OpenVINO/whisper-large-v3-int4-ov",
            Settings(),
            device="openvino:NPU",
            local_files_only=False,
        )
        backend._model_path = lambda: "/models/whisper"

        with (
            patch.dict("sys.modules", {"openvino_genai": fake_genai}),
            patch(
                "transclip.openvino_device.openvino_available_devices",
                return_value=("CPU", "NPU"),
            ),
        ):
            backend._load()

        self.assertEqual(captured["device"], "NPU")
        self.assertTrue(captured["kwargs"].get("STATIC_PIPELINE"))

    def test_openvino_load_skips_static_pipeline_on_cpu(self):
        captured = {}

        class FakeWhisperPipeline:
            def __init__(self, model_path, device, **kwargs):
                captured["device"] = device
                captured["kwargs"] = kwargs

        fake_genai = SimpleNamespace(WhisperPipeline=FakeWhisperPipeline)
        backend = OpenVINOWhisperBackend(
            "OpenVINO/whisper-large-v3-int4-ov",
            Settings(),
            device="openvino:CPU",
            local_files_only=False,
        )
        backend._model_path = lambda: "/models/whisper"

        with patch.dict("sys.modules", {"openvino_genai": fake_genai}):
            backend._load()

        self.assertEqual(captured["device"], "CPU")
        self.assertNotIn("STATIC_PIPELINE", captured["kwargs"])

    def test_openvino_missing_local_artifacts_raises(self):
        backend = OpenVINOWhisperBackend(
            "OpenVINO/whisper-large-v3-int4-ov",
            Settings(model_cache_dir="/transclip-nonexistent-cache"),
            local_files_only=True,
        )
        with self.assertRaisesRegex(RuntimeError, "prefetch"):
            backend._model_path()

    def test_whisper_language_token_formatting(self):
        self.assertEqual(_whisper_language_token("en"), "<|en|>")
        self.assertEqual(_whisper_language_token("<|fr|>"), "<|fr|>")
        self.assertEqual(_whisper_language_token(""), "")
        self.assertEqual(_whisper_language_token(None), "")

    def test_mlx_audio_reuses_loaded_model_object_across_transcriptions(self):
        loaded_model = object()
        load_calls = []
        generated_models = []
        generated_languages = []
        backend = MlxAudioASRBackend(
            "mlx-community/example",
            Settings(language="en"),
            local_files_only=False,
        )
        backend.audio_preparer = SimpleNamespace(
            prepare=lambda path: SimpleNamespace(wav_path=path, sample_rate=16000, temporary=False)
        )

        def fake_load_model(model_path):
            load_calls.append(model_path)
            return loaded_model

        def fake_generate(model, audio_path, output_stem, **kwargs):
            del audio_path, output_stem
            generated_models.append(model)
            generated_languages.append(kwargs["language"])
            return SimpleNamespace(text=f"transcript {len(generated_models)}")

        with (
            patch("transclip.asr.load_mlx_model", side_effect=fake_load_model),
            patch("transclip.asr.generate_transcription", side_effect=fake_generate),
        ):
            first = backend.transcribe(Path("first.wav"))
            second = backend.transcribe(Path("second.wav"))

        self.assertEqual(load_calls, ["mlx-community/example"])
        self.assertEqual(generated_models, [loaded_model, loaded_model])
        self.assertEqual(generated_languages, ["en", "en"])
        self.assertEqual(first.text, "transcript 1")
        self.assertEqual(second.text, "transcript 2")
        self.assertIn("model_load", second.timings_ms)
        self.assertIn("audio_prepare", second.timings_ms)
        self.assertIn("generate", second.timings_ms)

    def test_mlx_audio_preserves_empty_structured_transcript(self):
        backend = MlxAudioASRBackend(
            "mlx-community/example",
            Settings(language="en"),
            local_files_only=False,
        )
        backend.audio_preparer = SimpleNamespace(
            prepare=lambda path: SimpleNamespace(wav_path=path, sample_rate=16000, temporary=False)
        )

        with (
            patch("transclip.asr.load_mlx_model", return_value=object()),
            patch(
                "transclip.asr.generate_transcription",
                return_value=SimpleNamespace(text=""),
            ),
        ):
            result = backend.transcribe(Path("silent.wav"))

        self.assertEqual(result.text, "")

    def test_non_granite_model_is_rejected(self):
        with self.assertRaises(ValueError):
            build_asr_backend(Settings(asr_model="openai/whisper-tiny"))
        with self.assertRaises(ValueError):
            build_asr_backend(
                Settings(
                    asr_backend="granite_nar",
                    asr_model="ibm-granite/granite-speech-4.1-2b",
                )
            )
        with self.assertRaises(ValueError):
            build_asr_backend(
                Settings(
                    asr_backend="granite",
                    asr_model="ibm-granite/granite-speech-4.1-2b-nar",
                )
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

    def test_granite_ar_scales_max_new_tokens_with_audio_length(self):
        short = self._run_granite_ar_transcribe(audio_seconds=5)
        long = self._run_granite_ar_transcribe(audio_seconds=300)

        self.assertEqual(short["max_new_tokens"], 200)
        self.assertEqual(long["max_new_tokens"], 3064)

    def test_granite_ar_warns_when_generation_hits_token_cap(self):
        with self.assertLogs("transclip.asr", level="WARNING") as logs:
            result = self._run_granite_ar_transcribe(audio_seconds=5, output_tokens=200)

        self.assertEqual(result["max_new_tokens"], 200)
        self.assertIn("hit max_new_tokens=200", "\n".join(logs.output))

    def _run_granite_ar_transcribe(self, *, audio_seconds, output_tokens=10):
        class FakeModelInputs(dict):
            def to(self, device):
                del device
                return self

        class FakeTokens:
            def __init__(self, length):
                self.shape = (1, length)

            def unsqueeze(self, dim):
                del dim
                return self

        class FakeModelOutput:
            def __init__(self, length):
                self.length = length

            def __getitem__(self, item):
                row, token_slice = item
                del row, token_slice
                return FakeTokens(self.length)

        class FakeModel:
            def __init__(self):
                self.max_new_tokens = None

            def generate(self, **kwargs):
                self.max_new_tokens = kwargs["max_new_tokens"]
                return FakeModelOutput(output_tokens)

        class FakeTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                del args, kwargs
                return "templated"

            def batch_decode(self, *args, **kwargs):
                del args, kwargs
                return ["decoded transcript"]

        class FakeProcessor:
            def __call__(self, *args, **kwargs):
                del args, kwargs
                return FakeModelInputs({"input_ids": SimpleNamespace(shape=(1, 5))})

        model = FakeModel()
        backend = GraniteSpeechTransformersBackend("ibm-granite/granite-speech-4.1-2b")
        backend._device = lambda: "cpu"
        backend._loaded = (FakeProcessor(), FakeTokenizer(), model)
        backend.audio_preparer = SimpleNamespace(
            prepare=lambda _path: SimpleNamespace(
                wav=SimpleNamespace(shape=(1, int(audio_seconds * 16_000))),
                sample_rate=16_000,
            )
        )
        fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())

        with patch.dict("sys.modules", {"torch": fake_torch}):
            transcript = backend.transcribe(Path("audio.wav"))

        return {
            "max_new_tokens": model.max_new_tokens,
            "transcript": transcript.text,
        }

    def test_granite_nar_pads_waveform_to_stable_bucket(self):
        waveform = np.ones(int(1.2 * 16000), dtype=np.float32)

        padded = _pad_nar_waveform_to_bucket(waveform, sample_rate=16000)

        self.assertEqual(len(padded), 2 * 16000)
        np.testing.assert_allclose(padded[: len(waveform)], waveform)
        np.testing.assert_allclose(padded[len(waveform) :], 0.0)

    def test_granite_nar_bucket_padding_leaves_exact_bucket_unchanged(self):
        waveform = np.ones(2 * 16000, dtype=np.float32)

        padded = _pad_nar_waveform_to_bucket(waveform, sample_rate=16000)

        self.assertIs(padded, waveform)

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
            audio = DefaultASRAudioPreparer().prepare(Path("sample.wav"))

        self.assertEqual(audio.sample_rate, 16000)
        np.testing.assert_allclose(audio.wav.data, np.array([[2.0, 6.0]], dtype=np.float32))
        self.assertEqual(resample_calls, [(8000, 16000)])


if __name__ == "__main__":
    unittest.main()
