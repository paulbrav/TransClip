import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    validate_asr_model_backend,
)
from transclip.settings import Settings

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
            current = next(row for row in model_rows(settings, runtime) if row["model_id"] == settings.asr_model)
            text = next(row for row in model_rows(settings, runtime) if row["model_id"] == settings.text_model)
            self.assertEqual(current["marker"], "current,default")
            self.assertTrue(current["cached"])
            self.assertEqual(text["marker"], "current-text,default-text")

    def test_required_model_cache_paths_include_text_model_for_model_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                asr_model="local/asr",
                voice_model_cleanup_always_on=True,
                model_cache_dir=tmp,
            )

            self.assertEqual(
                required_model_cache_paths(settings),
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
            patch("transclip.models.shutil.disk_usage", return_value=usage),
            self.assertRaises(RuntimeError),
        ):
            settings = Settings(model_cache_dir=tmp)
            ensure_disk_space(settings, SUPPORTED_MODELS[0])


if __name__ == "__main__":
    unittest.main()
