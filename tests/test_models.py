import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from granite_speach.models import (
    SUPPORTED_MODELS,
    cache_artifacts_present,
    ensure_disk_space,
    model_rows,
    normalize_asr_backend,
    required_model_cache_paths,
    validate_asr_model_backend,
)
from granite_speach.settings import Settings


class ModelsTests(unittest.TestCase):
    def test_catalog_contains_current_granite_backends(self):
        rows = {(model.backend, model.model_id) for model in SUPPORTED_MODELS}

        self.assertIn(("granite_nar", "ibm-granite/granite-speech-4.1-2b-nar"), rows)
        self.assertIn(("granite", "ibm-granite/granite-speech-4.1-2b"), rows)

    def test_model_catalog_owns_asr_backend_compatibility(self):
        self.assertEqual(normalize_asr_backend("nar"), "granite_nar")
        self.assertEqual(
            validate_asr_model_backend("granite_nar", "ibm-granite/granite-speech-4.1-2b-nar"),
            "granite_nar",
        )
        self.assertEqual(
            validate_asr_model_backend("transformers", "ibm-granite/granite-speech-4.1-2b"),
            "granite",
        )
        with self.assertRaisesRegex(ValueError, "Granite NAR ASR requires"):
            validate_asr_model_backend("granite_nar", "ibm-granite/granite-speech-4.1-2b")
        with self.assertRaisesRegex(ValueError, "Use asr_backend"):
            validate_asr_model_backend("granite", "ibm-granite/granite-speech-4.1-2b-nar")

    def test_cache_detection_and_rows_do_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(model_cache_dir=tmp)
            model_dir = Path(tmp) / "models--ibm-granite--granite-speech-4.1-2b-nar" / "snapshots" / "abc"
            model_dir.mkdir(parents=True)

            self.assertTrue(cache_artifacts_present(settings.asr_model, settings))
            current = next(row for row in model_rows(settings) if row["model_id"] == settings.asr_model)
            self.assertEqual(current["marker"], "current,default")
            self.assertTrue(current["cached"])

    def test_required_model_cache_paths_include_cleanup_transformers(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                asr_model="local/asr",
                cleanup_model="local/cleanup",
                cleanup_runtime="transformers",
                model_cache_dir=tmp,
            )

            self.assertEqual(
                required_model_cache_paths(settings),
                [
                    Path(tmp) / "models--local--asr",
                    Path(tmp) / "models--local--cleanup",
                ],
            )

    def test_disk_space_failure_uses_catalog_estimate(self):
        usage = type("Usage", (), {"free": 1})()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("granite_speach.models.shutil.disk_usage", return_value=usage),
            self.assertRaises(RuntimeError),
        ):
            settings = Settings(model_cache_dir=tmp)
            ensure_disk_space(settings, SUPPORTED_MODELS[0])


if __name__ == "__main__":
    unittest.main()
