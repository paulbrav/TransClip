import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.platform_runtime import DefaultPlatformRuntime
from transclip.settings import (
    DEFAULT_HOTKEY_LINUX,
    Settings,
    coerce_setting_value,
    load_settings,
    patch_settings,
    set_setting,
    write_default_settings,
)

from tests.service_helpers import FakeRuntime


class SettingsTests(unittest.TestCase):
    def test_default_files_round_trip(self):
        runtime = FakeRuntime(system="Linux", home=Path("/home/user"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_file = write_default_settings(root / "settings.toml", runtime=runtime)

            settings = load_settings(settings_file, runtime=runtime)

            self.assertEqual(settings.hotkey_linux, DEFAULT_HOTKEY_LINUX)
            self.assertEqual(settings.max_recording_seconds, 60)
            self.assertEqual(settings.toggle_cooldown_ms, 500)
            self.assertEqual(settings.asr_backend, "granite_nar")
            self.assertEqual(settings.asr_model, "ibm-granite/granite-speech-4.1-2b-nar")
            self.assertTrue(settings.voice_mode_routing_enabled)
            self.assertFalse(settings.voice_model_cleanup_always_on)
            self.assertTrue(settings.voice_mode_shell_enabled)
            self.assertEqual(settings.text_model_runtime, "transformers")
            self.assertEqual(settings.text_model, "Qwen/Qwen3.5-4B")
            self.assertTrue(settings.shell_syntax_validation_enabled)
            self.assertTrue(settings.shellcheck_enabled)
            self.assertTrue(settings.models_local_files_only)
            self.assertEqual(settings.model_cache_dir, "")

    def test_unknown_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            path.write_text('hotkey_linux = "Ctrl+Space"\nwat = true\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(path)

    def test_platform_helpers_have_defaults(self):
        with patch.object(DefaultPlatformRuntime, "system", return_value="Linux"):
            settings = Settings()
            self.assertIn("XF86TouchpadOff", settings.active_hotkey)
            self.assertIn("V", settings.paste_shortcut)

    def test_active_hotkey_uses_macos_binding_on_darwin(self):
        with patch.object(DefaultPlatformRuntime, "system", return_value="Darwin"):
            settings = Settings()
            self.assertEqual(settings.active_hotkey, "Option+Space")
            self.assertNotIn("XF86TouchpadOff", settings.active_hotkey)
            self.assertEqual(settings.paste_shortcut, "Command+V")

    def test_patch_settings_returns_new_object_without_mutating_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            write_default_settings(path)
            original = load_settings(path)

            updated = patch_settings(path, toggle_cooldown_ms=750)

            self.assertEqual(updated.toggle_cooldown_ms, 750)
            self.assertEqual(original.toggle_cooldown_ms, 500)
            self.assertEqual(load_settings(path).toggle_cooldown_ms, 750)

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
        self.assertIs(coerce_setting_value("voice_model_cleanup_always_on", "on"), True)
        self.assertEqual(coerce_setting_value("sample_rate", "22050"), 22050)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            write_default_settings(path)
            with self.assertRaises(ValueError):
                set_setting(path, "wat", "true")


if __name__ == "__main__":
    unittest.main()
