import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from transclip.models import (
    model_rows,
    supported_catalog_entries,
    validate_asr_model_backend,
)
from transclip.platform_runtime import user_cache_dir, user_config_dir, user_log_dir
from transclip.runtime_profile import detect_runtime_profile
from transclip.settings import Settings, default_settings

from tests.service_helpers import FakeRuntime


class PlatformRuntimeTests(unittest.TestCase):
    def test_linux_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Linux", home=Path(tmp))
            self.assertEqual(user_config_dir("transclip", runtime), Path(tmp) / ".config" / "transclip")
            self.assertEqual(user_cache_dir("transclip", runtime), Path(tmp) / ".cache" / "transclip")
            self.assertEqual(user_log_dir("transclip", runtime), Path(tmp) / ".cache" / "transclip")

    def test_darwin_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            self.assertEqual(
                user_config_dir("transclip", runtime),
                Path(tmp) / "Library" / "Application Support" / "transclip",
            )
            self.assertEqual(
                user_cache_dir("transclip", runtime),
                Path(tmp) / "Library" / "Caches" / "transclip",
            )
            self.assertEqual(user_log_dir("transclip", runtime), Path(tmp) / "Library" / "Logs" / "transclip")

    def test_linux_profile_defaults(self):
        runtime = FakeRuntime(system="Linux", home=Path("/home/user"))
        profile = detect_runtime_profile(runtime)
        settings = default_settings(runtime)

        self.assertEqual(profile.profile_id, "linux_gpu")
        self.assertEqual(settings.asr_backend, "granite_nar")
        self.assertEqual(settings.asr_model, "ibm-granite/granite-speech-4.1-2b-nar")

    def test_linux_cpu_profile_defaults_to_autoregressive_granite(self):
        runtime = FakeRuntime(system="Linux", home=Path("/home/user"))
        with patch("transclip.runtime_profile.machine_architecture", return_value="armv7l"):
            profile = detect_runtime_profile(runtime)
            settings = default_settings(runtime)

        self.assertEqual(profile.profile_id, "linux_cpu")
        self.assertEqual(settings.asr_backend, "granite")
        self.assertEqual(settings.asr_model, "ibm-granite/granite-speech-4.1-2b")
        self.assertEqual(settings.asr_device, "cpu")

    def test_darwin_arm_profile_defaults(self):
        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="arm64")
        profile = detect_runtime_profile(runtime)
        settings = default_settings(runtime)

        self.assertEqual(profile.profile_id, "darwin_arm_mlx")
        self.assertEqual(settings.asr_backend, "mlx_audio_whisper")
        self.assertEqual(settings.asr_model, "mlx-community/whisper-large-v3-turbo-asr-fp16")
        self.assertEqual(settings.asr_device, "auto")

    def test_darwin_non_arm_profile_defaults_to_valid_file_backend(self):
        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="x86_64")
        settings = default_settings(runtime)

        self.assertEqual(settings.asr_backend, "file:/dev/null")
        self.assertEqual(settings.asr_model, "")

    def test_granite_nar_rejected_on_darwin_arm(self):
        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="arm64")
        settings = replace(
            default_settings(runtime),
            asr_backend="granite_nar",
            asr_model="ibm-granite/granite-speech-4.1-2b-nar",
        )
        with self.assertRaisesRegex(ValueError, "not supported on Darwin"):
            validate_asr_model_backend(settings.asr_backend, settings.asr_model, runtime)

    def test_supported_catalog_entries_filter_by_platform(self):
        linux_runtime = FakeRuntime(system="Linux", home=Path("/home/user"))
        darwin_runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="arm64")

        linux_ids = {entry.model_id for entry in supported_catalog_entries(linux_runtime)}
        darwin_ids = {entry.model_id for entry in supported_catalog_entries(darwin_runtime)}

        self.assertIn("ibm-granite/granite-speech-4.1-2b-nar", linux_ids)
        self.assertNotIn("ibm-granite/granite-speech-4.1-2b-nar", darwin_ids)
        self.assertIn("mlx-community/whisper-large-v3-turbo-asr-fp16", darwin_ids)
        self.assertNotIn("mlx-community/whisper-large-v3-turbo-asr-fp16", linux_ids)

    def test_model_rows_include_text_models(self):
        runtime = FakeRuntime(system="Linux", home=Path("/home/user"))
        settings = Settings()
        rows = model_rows(settings, runtime)
        backends = {row["backend"] for row in rows}
        self.assertIn("text_generation", backends)


if __name__ == "__main__":
    unittest.main()
