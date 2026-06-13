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
    PathAudioPreparer,
    _configure_rocm_nar_attention_env,
    _granite_nar_dtype,
    _pad_nar_waveform_to_bucket,
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

    def test_mlx_audio_reuses_loaded_model_object_across_transcriptions(self):
        loaded_model = SimpleNamespace(dims=SimpleNamespace(n_text_ctx=448))
        load_calls = []
        generated_models = []
        generated_options = []
        prepared_durations = iter((2.0, 16.0))
        backend = MlxAudioASRBackend(
            "mlx-community/example",
            Settings(language="en"),
            local_files_only=False,
        )
        backend.audio_preparer = SimpleNamespace(
            prepare=lambda path: SimpleNamespace(
                wav_path=path,
                sample_rate=16000,
                duration_seconds=next(prepared_durations),
                temporary=False,
            )
        )

        def fake_load_model(model_path):
            load_calls.append(model_path)
            return loaded_model

        def fake_generate(model, audio_path, output_stem, **kwargs):
            del audio_path, output_stem
            generated_models.append(model)
            generated_options.append(kwargs)
            return SimpleNamespace(text=f"transcript {len(generated_models)}")

        with (
            patch("transclip.asr.load_mlx_model", side_effect=fake_load_model),
            patch("transclip.asr.generate_transcription", side_effect=fake_generate),
        ):
            first = backend.transcribe(Path("first.wav"))
            second = backend.transcribe(Path("second.wav"))

        self.assertEqual(load_calls, ["mlx-community/example"])
        self.assertEqual(generated_models, [loaded_model, loaded_model])
        self.assertEqual([item["language"] for item in generated_options], ["en", "en"])
        self.assertEqual([item["temperature"] for item in generated_options], [0.0, 0.0])
        self.assertEqual([item["return_timestamps"] for item in generated_options], [False, False])
        self.assertEqual([item["condition_on_previous_text"] for item in generated_options], [False, False])
        self.assertEqual([item["sample_len"] for item in generated_options], [48, 128])
        self.assertEqual(first.text, "transcript 1")
        self.assertEqual(second.text, "transcript 2")
        self.assertIn("model_load", second.timings_ms)
        self.assertIn("audio_prepare", second.timings_ms)
        self.assertIn("generate_write", second.timings_ms)

    def test_mlx_audio_uses_padded_duration_for_decode_shape(self):
        loaded_model = SimpleNamespace(dims=SimpleNamespace(n_text_ctx=448))
        generated_options = []
        backend = MlxAudioASRBackend(
            "mlx-community/example",
            Settings(language="en"),
            local_files_only=False,
        )
        backend.audio_preparer = SimpleNamespace(
            prepare=lambda path: SimpleNamespace(
                wav_path=path,
                sample_rate=16000,
                duration_seconds=27.3,
                padded_duration_seconds=28.0,
                temporary=False,
            )
        )

        def fake_generate(model, audio_path, output_stem, **kwargs):
            del model, audio_path, output_stem
            generated_options.append(kwargs)
            return SimpleNamespace(text="transcript")

        with (
            patch("transclip.asr.load_mlx_model", return_value=loaded_model),
            patch("transclip.asr.generate_transcription", side_effect=fake_generate),
        ):
            backend.transcribe(Path("audio.wav"))

        self.assertEqual(generated_options[0]["sample_len"], 200)

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

    def test_path_audio_preparer_trims_edge_silence_before_writing_temp_wav(self):
        samples = np.array([[0.0]] * 5 + [[0.1]] * 4 + [[0.0]] * 5, dtype=np.float32)
        writes = []
        fake_soundfile = SimpleNamespace(
            read=lambda *_args, **_kwargs: (samples, 10),
            write=lambda path, data, sample_rate: writes.append(
                (Path(path), np.array(data), sample_rate)
            ),
        )

        with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
            audio = PathAudioPreparer(target_sample_rate=10).prepare(Path("sample.wav"))

        self.assertTrue(audio.temporary)
        self.assertEqual(audio.sample_rate, 10)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][2], 10)
        np.testing.assert_allclose(
            writes[0][1],
            np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0], dtype=np.float32),
        )

    def test_path_audio_preparer_pads_to_stable_audio_bucket_without_extending_duration(self):
        samples = np.array([[0.1]] * 25, dtype=np.float32)
        writes = []
        fake_soundfile = SimpleNamespace(
            read=lambda *_args, **_kwargs: (samples, 10),
            write=lambda path, data, sample_rate: writes.append(
                (Path(path), np.array(data), sample_rate)
            ),
        )

        with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
            audio = PathAudioPreparer(
                target_sample_rate=10,
                bucket_seconds=4.0,
                minimum_seconds=4.0,
            ).prepare(Path("sample.wav"))

        self.assertTrue(audio.temporary)
        self.assertEqual(audio.sample_rate, 10)
        self.assertEqual(audio.duration_seconds, 2.5)
        self.assertEqual(audio.padded_duration_seconds, 4.0)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][2], 10)
        self.assertEqual(len(writes[0][1]), 40)
        np.testing.assert_allclose(writes[0][1][:25], 0.1)
        np.testing.assert_allclose(writes[0][1][25:], 0.0)

    def test_path_audio_preparer_uses_one_second_buckets_for_short_audio(self):
        samples = np.array([[0.1]] * 25, dtype=np.float32)
        writes = []
        fake_soundfile = SimpleNamespace(
            read=lambda *_args, **_kwargs: (samples, 10),
            write=lambda path, data, sample_rate: writes.append(
                (Path(path), np.array(data), sample_rate)
            ),
        )

        with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
            audio = PathAudioPreparer(
                target_sample_rate=10,
                bucket_seconds=4.0,
                minimum_seconds=1.0,
                short_bucket_seconds=1.0,
                short_bucket_max_seconds=12.0,
            ).prepare(Path("sample.wav"))

        self.assertEqual(audio.duration_seconds, 2.5)
        self.assertEqual(audio.padded_duration_seconds, 3.0)
        self.assertEqual(len(writes[0][1]), 30)
        np.testing.assert_allclose(writes[0][1][:25], 0.1)
        np.testing.assert_allclose(writes[0][1][25:], 0.0)

    def test_path_audio_preparer_uses_long_buckets_above_short_range(self):
        samples = np.array([[0.1]] * 125, dtype=np.float32)
        writes = []
        fake_soundfile = SimpleNamespace(
            read=lambda *_args, **_kwargs: (samples, 10),
            write=lambda path, data, sample_rate: writes.append(
                (Path(path), np.array(data), sample_rate)
            ),
        )

        with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
            audio = PathAudioPreparer(
                target_sample_rate=10,
                bucket_seconds=4.0,
                minimum_seconds=1.0,
                short_bucket_seconds=1.0,
                short_bucket_max_seconds=12.0,
            ).prepare(Path("sample.wav"))

        self.assertEqual(audio.duration_seconds, 12.5)
        self.assertEqual(audio.padded_duration_seconds, 16.0)
        self.assertEqual(len(writes[0][1]), 160)


if __name__ == "__main__":
    unittest.main()
