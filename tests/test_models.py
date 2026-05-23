import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from transclip.cli import main
from transclip.models import (
    MODEL_CATALOG,
    SUPPORTED_MODELS,
    SUPPORTED_TEXT_MODELS,
    cache_artifacts_present,
    ensure_disk_space,
    model_rows,
    normalize_asr_backend,
    prefetch_model,
    required_model_cache_paths,
    supported_catalog_entries,
    validate_asr_model_backend,
)
from transclip.settings import Settings, default_settings, write_settings

from tests.service_helpers import FakeRuntime, linux_gpu_runtime, patch_linux_gpu_runtime


class ModelsTests(unittest.TestCase):
    @staticmethod
    def _linux_runtime() -> FakeRuntime:
        return FakeRuntime(system="Linux", home=Path("/home/user"))

    def test_catalog_contains_current_granite_backends(self):
        rows = {(model.backend, model.model_id) for model in SUPPORTED_MODELS}

        self.assertIn(("granite_nar", "ibm-granite/granite-speech-4.1-2b-nar"), rows)
        self.assertIn(("granite", "ibm-granite/granite-speech-4.1-2b"), rows)
        self.assertGreater(len(MODEL_CATALOG), 2)
        text_rows = {(model.backend, model.model_id) for model in SUPPORTED_TEXT_MODELS}
        self.assertIn(("text_generation", "Qwen/Qwen3.5-4B"), text_rows)

    def test_model_display_name_comes_from_catalog(self):
        from transclip.models import model_display_name

        self.assertEqual(
            model_display_name("ibm-granite/granite-speech-4.1-2b"),
            "Keyword-biased ASR - Granite 4.1",
        )

    def test_model_catalog_owns_asr_backend_compatibility(self):
        runtime = self._linux_runtime()
        self.assertEqual(normalize_asr_backend("nar"), "granite_nar")
        self.assertEqual(
            validate_asr_model_backend("granite_nar", "ibm-granite/granite-speech-4.1-2b-nar", runtime),
            "granite_nar",
        )
        self.assertEqual(
            validate_asr_model_backend("transformers", "ibm-granite/granite-speech-4.1-2b", runtime),
            "granite",
        )
        with self.assertRaisesRegex(ValueError, "requires asr_backend='granite'"):
            validate_asr_model_backend("granite_nar", "ibm-granite/granite-speech-4.1-2b", runtime)
        with self.assertRaisesRegex(ValueError, "requires asr_backend='granite_nar'"):
            validate_asr_model_backend("granite", "ibm-granite/granite-speech-4.1-2b-nar", runtime)

    def test_cache_detection_and_rows_do_not_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch_linux_gpu_runtime():
            runtime = linux_gpu_runtime()
            settings = Settings(model_cache_dir=tmp)
            model_dir = Path(tmp) / "models--ibm-granite--granite-speech-4.1-2b-nar" / "snapshots" / "abc"
            model_dir.mkdir(parents=True)

            self.assertTrue(cache_artifacts_present(settings.asr_model, settings))
            current = next(row for row in model_rows(settings, runtime) if row.model_id == settings.asr_model)
            text = next(row for row in model_rows(settings, runtime) if row.model_id == settings.text_model)
            self.assertEqual(current.marker, "current,default")
            self.assertTrue(current.cached)
            self.assertEqual(text.marker, "current-text,default-text")

    def test_required_model_cache_paths_include_text_model_for_model_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                asr_model="local/asr",
                voice_model_cleanup_always_on=True,
                model_cache_dir=tmp,
            )

            self.assertEqual(
                required_model_cache_paths(
                    settings,
                    extra_model_ids=(settings.text_model,),
                ),
                [
                    Path(tmp) / "models--local--asr",
                    Path(tmp) / "models--Qwen--Qwen3.5-4B",
                ],
            )

    def test_prefetch_text_model_uses_image_text_to_text_contract(self):
        calls = []

        class AutoModelForImageTextToText:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append(("model", args, kwargs))

        class AutoProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append(("processor", args, kwargs))

        transformers = type(
            "TransformersModule",
            (),
            {
                "AutoModelForImageTextToText": AutoModelForImageTextToText,
                "AutoProcessor": AutoProcessor,
            },
        )()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(sys.modules, {"transformers": transformers}),
            patch("transclip.models.ensure_disk_space"),
        ):
            path = prefetch_model("Qwen/Qwen3.5-4B", Settings(model_cache_dir=tmp))

        self.assertEqual(path, Path(tmp) / "models--Qwen--Qwen3.5-4B")
        self.assertEqual([call[0] for call in calls], ["model", "processor"])

    def test_disk_space_failure_uses_catalog_estimate(self):
        usage = type("Usage", (), {"free": 1})()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("transclip.models.cache.shutil.disk_usage", return_value=usage),
            self.assertRaises(RuntimeError),
        ):
            settings = Settings(model_cache_dir=tmp)
            ensure_disk_space(settings, SUPPORTED_MODELS[0])

    def test_windows_catalog_excludes_nar_and_defaults_to_granite_ar(self):
        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))
        with patch("transclip.device.torch_cuda_usable", return_value=True):
            defaults = default_settings(runtime)
            entries = supported_catalog_entries(runtime)
            model_ids = {entry.model_id for entry in entries}
            rows = model_rows(defaults, runtime)

        self.assertEqual(defaults.asr_backend, "granite")
        self.assertEqual(defaults.asr_model, "ibm-granite/granite-speech-4.1-2b")
        self.assertIn("ibm-granite/granite-speech-4.1-2b", model_ids)
        self.assertIn("ibm-granite/granite-speech-4.1-2b-plus", model_ids)
        self.assertNotIn("ibm-granite/granite-speech-4.1-2b-nar", model_ids)
        default_row = next(row for row in rows if "default" in row.marker)
        self.assertEqual(default_row.backend, "granite")
        self.assertEqual(default_row.model_id, "ibm-granite/granite-speech-4.1-2b")

    def test_windows_rejects_granite_nar_backend(self):
        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))
        with (
            patch("transclip.device.torch_cuda_usable", return_value=True),
            self.assertRaisesRegex(ValueError, "not supported on Windows"),
        ):
            validate_asr_model_backend(
                "granite_nar",
                "ibm-granite/granite-speech-4.1-2b-nar",
                runtime,
            )

    def test_cli_models_list_uses_local_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.toml"
            write_settings(Settings(host="127.0.0.1", port=0), settings_path)
            stdout = io.StringIO()
            with patch_linux_gpu_runtime(), redirect_stdout(stdout):
                code = main(["--settings", str(settings_path), "models", "list"])

            self.assertEqual(code, 0)
            self.assertIn("ibm-granite/granite-speech-4.1-2b-nar", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
