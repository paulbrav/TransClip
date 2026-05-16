import unittest

from granite_speach.asr import (
    FileTranscriptASRBackend,
    GraniteSpeechNarTransformersBackend,
    GraniteSpeechTransformersBackend,
    build_asr_backend,
    granite_user_prompt,
)
from granite_speach.settings import Settings


class ASRTests(unittest.TestCase):
    def test_granite_keyword_prompt_matches_model_card_shape(self):
        prompt = granite_user_prompt(["PyTorch", "ROCm"])
        self.assertEqual(prompt, "transcribe the speech to text. Keywords: PyTorch, ROCm")

    def test_granite_prompt_without_keywords_requests_punctuation(self):
        self.assertEqual(
            granite_user_prompt([]),
            "transcribe the speech with proper punctuation and capitalization.",
        )

    def test_backend_selection(self):
        backend = build_asr_backend(Settings(model_cache_dir="/models"))
        self.assertIsInstance(backend, GraniteSpeechNarTransformersBackend)
        self.assertTrue(backend.local_files_only)
        self.assertEqual(backend.cache_dir, "/models")
        ar_backend = build_asr_backend(
            Settings(
                asr_backend="granite",
                asr_model="ibm-granite/granite-speech-4.1-2b",
            )
        )
        self.assertIsInstance(ar_backend, GraniteSpeechTransformersBackend)
        self.assertIsInstance(
            build_asr_backend(Settings(asr_backend="file:/tmp/transcript.txt")),
            FileTranscriptASRBackend,
        )
        nar_backend = build_asr_backend(
            Settings(
                asr_backend="granite_nar",
                asr_model="ibm-granite/granite-speech-4.1-2b-nar",
                model_cache_dir="/models",
            )
        )
        self.assertIsInstance(nar_backend, GraniteSpeechNarTransformersBackend)
        self.assertTrue(nar_backend.local_files_only)
        self.assertEqual(nar_backend.cache_dir, "/models")

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


if __name__ == "__main__":
    unittest.main()
