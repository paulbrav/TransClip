import tempfile
import unittest
from pathlib import Path

from transclip.settings import (
    DEFAULT_HOTKEY_LINUX,
    Settings,
    coerce_setting_value,
    load_settings,
    set_setting,
    write_default_settings,
)


class SettingsTests(unittest.TestCase):
    def test_default_files_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_file = write_default_settings(root / "settings.toml")

            settings = load_settings(settings_file)

            self.assertEqual(settings.hotkey_linux, DEFAULT_HOTKEY_LINUX)
            self.assertEqual(settings.max_recording_seconds, 60)
            self.assertEqual(settings.toggle_cooldown_ms, 500)
            self.assertEqual(settings.asr_backend, "granite_nar")
            self.assertEqual(settings.asr_model, "ibm-granite/granite-speech-4.1-2b-nar")
            self.assertEqual(settings.cleanup_runtime, "rule")
            self.assertEqual(settings.cleanup_model, "google/gemma-4-E2B-it")
            self.assertTrue(settings.models_local_files_only)
            self.assertEqual(settings.model_cache_dir, "")

    def test_unknown_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            path.write_text('hotkey_linux = "Ctrl+Space"\nwat = true\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(path)

    def test_platform_helpers_have_defaults(self):
        settings = Settings()
        hotkey = settings.active_hotkey
        if "XF86TouchpadOff" in settings.hotkey_linux:
            self.assertTrue("XF86TouchpadOff" in hotkey or "Option+Space" in hotkey)
        else:
            self.assertTrue(hotkey)
        self.assertIn("V", settings.paste_shortcut)

    def test_set_setting_rewrites_canonical_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            write_default_settings(path)

            updated = set_setting(path, "toggle_cooldown_ms", "750")

            self.assertEqual(updated.toggle_cooldown_ms, 750)
            self.assertEqual(load_settings(path).toggle_cooldown_ms, 750)
            self.assertIn("toggle_cooldown_ms = 750", path.read_text(encoding="utf-8"))

    def test_setting_type_coercion_and_unknown_field(self):
        self.assertIs(coerce_setting_value("cleanup_enabled", "false"), False)
        self.assertEqual(coerce_setting_value("sample_rate", "22050"), 22050)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            write_default_settings(path)
            with self.assertRaises(ValueError):
                set_setting(path, "wat", "true")


if __name__ == "__main__":
    unittest.main()
