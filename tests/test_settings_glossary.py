from pathlib import Path
import tempfile
import unittest

from granite_speach.glossary import keyword_prompt, load_keywords
from granite_speach.settings import Settings, load_settings, write_default_keywords, write_default_settings


class SettingsGlossaryTests(unittest.TestCase):
    def test_default_files_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_file = write_default_settings(root / "settings.toml")
            keywords_file = write_default_keywords(root / "keywords.txt")

            settings = load_settings(settings_file)
            terms = load_keywords(keywords_file)

            self.assertEqual(settings.hotkey_linux, "<Super><Shift>XF86TouchpadOff")
            self.assertEqual(settings.max_recording_seconds, 60)
            self.assertEqual(settings.asr_backend, "granite_nar")
            self.assertEqual(settings.asr_model, "ibm-granite/granite-speech-4.1-2b-nar")
            self.assertEqual(settings.cleanup_runtime, "rule")
            self.assertEqual(settings.cleanup_model, "google/gemma-4-E2B-it")
            self.assertTrue(settings.models_local_files_only)
            self.assertEqual(settings.model_cache_dir, "")
            self.assertIn("ROCm", terms)
            self.assertIn("Granite", keyword_prompt(terms))

    def test_unknown_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            path.write_text('hotkey_linux = "Ctrl+Space"\nwat = true\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(path)

    def test_platform_helpers_have_defaults(self):
        settings = Settings()
        self.assertIn("XF86TouchpadOff", settings.active_hotkey)
        self.assertIn("V", settings.paste_shortcut)


if __name__ == "__main__":
    unittest.main()
