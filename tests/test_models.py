import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.platform_helpers import darwin_arm_runtime, linux_runtime
from transclip.models import (
    SUPPORTED_MODELS,
    cache_artifacts_present,
    ensure_disk_space,
    mlx_snapshot_path,
    model_rows,
    normalize_asr_backend,
    prefetch_model,
    required_model_cache_paths,
    validate_asr_model_backend,
)
from transclip.settings import Settings


class ModelsTests(unittest.TestCase):
    linux = linux_runtime()

    def test_catalog_contains_current_granite_backends(self):
        rows = {(model.backend, model.model_id) for model in SUPPORTED_MODELS}

        self.assertIn(("granite_nar", "ibm-granite/granite-speech-4.1-2b-nar"), rows)
        self.assertIn(("granite", "ibm-granite/granite-speech-4.1-2b"), rows)
        self.assertIn(("mlx_audio_whisper", "mlx-community/whisper-large-v3-turbo-asr-fp16"), rows)
        self.assertIn(("granite_mlx", "mlx-community/granite-4.0-1b-speech-8bit"), rows)

    def test_mlx_aliases_normalize(self):
        self.assertEqual(normalize_asr_backend("mlx"), "mlx_audio_whisper")
        self.assertEqual(normalize_asr_backend("mlx_whisper"), "mlx_audio_whisper")

    def test_model_catalog_owns_asr_backend_compatibility(self):
        self.assertEqual(normalize_asr_backend("nar"), "granite_nar")
        self.assertEqual(
            validate_asr_model_backend(
                "granite_nar",
                "ibm-granite/granite-speech-4.1-2b-nar",
                self.linux,
            ),
            "granite_nar",
        )
        self.assertEqual(
            validate_asr_model_backend(
                "transformers",
                "ibm-granite/granite-speech-4.1-2b",
                self.linux,
            ),
            "granite",
        )
        with self.assertRaisesRegex(ValueError, "requires asr_backend='granite'"):
            validate_asr_model_backend("granite_nar", "ibm-granite/granite-speech-4.1-2b", self.linux)
        with self.assertRaisesRegex(ValueError, "requires asr_backend='granite_nar'"):
            validate_asr_model_backend("granite", "ibm-granite/granite-speech-4.1-2b-nar", self.linux)

    def test_cache_detection_and_rows_do_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(model_cache_dir=tmp)
            model_dir = Path(tmp) / "models--ibm-granite--granite-speech-4.1-2b-nar" / "snapshots" / "abc"
            model_dir.mkdir(parents=True)

            self.assertTrue(cache_artifacts_present(settings.asr_model, settings))
            current = next(
                row for row in model_rows(settings, runtime=self.linux) if row["model_id"] == settings.asr_model
            )
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

    def test_mlx_prefetch_uses_platform_cache_root(self):
        captured: dict[str, str] = {}

        def fake_snapshot_download(**kwargs):
            captured.update(kwargs)
            return "/tmp/fake"

        fake_huggingface_hub = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
        with patch.dict("sys.modules", {"huggingface_hub": fake_huggingface_hub}):
            prefetch_model(
                "mlx-community/whisper-large-v3-turbo-asr-fp16",
                Settings(),
                darwin_arm_runtime(),
            )

        self.assertIn("Library/Caches/huggingface/hub", captured["cache_dir"])

    def test_mlx_snapshot_path_uses_refs_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(model_cache_dir=tmp)
            root = Path(tmp) / "models--mlx-community--whisper-large-v3-turbo-asr-fp16"
            (root / "snapshots" / "aaa").mkdir(parents=True)
            selected = root / "snapshots" / "bbb"
            selected.mkdir()
            (root / "snapshots" / "zzz").mkdir()
            (root / "refs").mkdir()
            (root / "refs" / "main").write_text("bbb\n", encoding="utf-8")

            self.assertEqual(
                mlx_snapshot_path("mlx-community/whisper-large-v3-turbo-asr-fp16", settings),
                selected,
            )

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
