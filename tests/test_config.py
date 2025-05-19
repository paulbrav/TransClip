import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_dir = Path(self.temp_dir.name) / "transclip"
        self.config_file = self.config_dir / "config.json"

        # Stub transclip.app to avoid heavy imports when loading the package
        app_module = types.ModuleType("transclip.app")
        app_module.WhisperModelType = type("WhisperModelType", (), {})
        self.app_patch = mock.patch.dict(sys.modules, {"transclip.app": app_module})
        self.app_patch.start()
        self.addCleanup(self.app_patch.stop)

        import transclip.config as config_module
        importlib.reload(config_module)
        self.config = config_module

        patcher1 = mock.patch.object(self.config, "CONFIG_DIR", self.config_dir)
        patcher2 = mock.patch.object(self.config, "CONFIG_FILE", self.config_file)
        patcher1.start()
        patcher2.start()
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)

    def test_load_config_missing(self):
        self.assertEqual(self.config.load_config(), {})

    def test_load_config_invalid(self):
        self.config_dir.mkdir(parents=True)
        self.config_file.write_text("{invalid", encoding="utf-8")
        self.assertEqual(self.config.load_config(), {})

    def test_save_and_load(self):
        data = {"recording_key": "Key.home"}
        self.config.save_config(data)
        self.assertTrue(self.config_file.exists())
        self.assertEqual(self.config.load_config(), data)


if __name__ == "__main__":
    unittest.main()
