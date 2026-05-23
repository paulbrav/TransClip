import unittest
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
    _configure_rocm_nar_attention_env,
    _granite_nar_dtype,
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
