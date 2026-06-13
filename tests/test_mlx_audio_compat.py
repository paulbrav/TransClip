import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.mlx_audio_compat import generate_transcription


class MlxAudioCompatTests(unittest.TestCase):
    def test_generate_transcription_preserves_decode_options_for_loaded_models(self):
        model = FakeLoadedModel()
        module = types.ModuleType("mlx_audio.stt.generate")
        module.generate_transcription = fake_mlx_generate_transcription

        with patch.dict(
            sys.modules,
            {
                "mlx_audio": types.ModuleType("mlx_audio"),
                "mlx_audio.stt": types.ModuleType("mlx_audio.stt"),
                "mlx_audio.stt.generate": module,
            },
        ):
            result = generate_transcription(
                model,
                Path("audio.wav"),
                "transcript",
                language="en",
                temperature=0.0,
                sample_len=48,
            )

        self.assertEqual(result.text, "ok")
        self.assertEqual(model.generate_kwargs["language"], "en")
        self.assertEqual(model.generate_kwargs["temperature"], 0.0)
        self.assertEqual(model.generate_kwargs["sample_len"], 48)


class FakeLoadedModel:
    def __init__(self):
        self.generate_kwargs = {}

    def generate(
        self,
        audio,
        *,
        verbose=None,
        language=None,
        temperature=(0.0, 0.2),
        **decode_options,
    ):
        self.generate_kwargs = {
            "audio": audio,
            "verbose": verbose,
            "language": language,
            "temperature": temperature,
            **decode_options,
        }
        return types.SimpleNamespace(text="ok")


def fake_mlx_generate_transcription(
    model=None,
    audio=None,
    output_path="transcript",
    format="txt",
    verbose=False,
    **kwargs,
):
    del output_path, format
    signature = inspect.signature(model.generate)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return model.generate(audio, verbose=verbose, **filtered)


if __name__ == "__main__":
    unittest.main()
