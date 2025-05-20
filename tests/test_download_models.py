import importlib
import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path


class DownloadModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Stub faster_whisper module
        fw_module = types.ModuleType("faster_whisper")
        fw_module.download_model = lambda *args, **kwargs: None

        # Stub transclip.transcription module with minimal WhisperModelType
        app_module = types.ModuleType("transclip.transcription")
        from enum import Enum

        class WhisperModelType(str, Enum):
            TINY = "tiny"
            BASE = "base"
            SMALL = "small"
            MEDIUM = "medium"
            LARGE = "large"
            LARGE_V2 = "large-v2"
            LARGE_V3 = "large-v3"
            PARAKEET_TDT_0_6B_V2 = "nvidia/parakeet-tdt-0.6b-v2"

            @classmethod
            def get_description(cls, model_type: "WhisperModelType") -> str:
                return {
                    cls.TINY: "Tiny",
                    cls.BASE: "Base",
                    cls.SMALL: "Small",
                    cls.MEDIUM: "Medium",
                    cls.LARGE: "Large",
                    cls.LARGE_V2: "Large-v2",
                    cls.LARGE_V3: "Large-v3",
                    cls.PARAKEET_TDT_0_6B_V2: "Parakeet",
                }[model_type]

        app_module.WhisperModelType = WhisperModelType

        self.module_patch = unittest.mock.patch.dict(sys.modules, {
            "faster_whisper": fw_module,
            "transclip.transcription": app_module,
        })
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

        import transclip.download_models as dm
        importlib.reload(dm)
        self.dm = dm
        self.WhisperModelType = dm.WhisperModelType

    def test_get_model_size_mb(self):
        self.assertEqual(self.dm.get_model_size_mb(self.WhisperModelType.TINY), 75)
        self.assertEqual(self.dm.get_model_size_mb(self.WhisperModelType.PARAKEET_TDT_0_6B_V2), 1200)

    def test_check_disk_space(self):
        usage = namedtuple('usage', ['total', 'used', 'free'])
        # Enough space
        def du(_):
            return usage(100, 0, 200 * 1024 * 1024)
        with unittest.mock.patch.object(self.dm.shutil, 'disk_usage', du):
            self.assertTrue(self.dm.check_disk_space(100))
        # Not enough
        def du2(_):
            return usage(100, 0, 50 * 1024 * 1024)
        with unittest.mock.patch.object(self.dm.shutil, 'disk_usage', du2):
            self.assertFalse(self.dm.check_disk_space(100))

    def test_download_model_already_exists(self):
        called = False

        def fake_download(*args, **kwargs):
            nonlocal called
            called = True

        with unittest.mock.patch.object(self.dm, 'download_model', side_effect=fake_download):
            with unittest.mock.patch.object(Path, 'exists', return_value=True):
                with unittest.mock.patch.object(self.dm, 'check_disk_space', return_value=True):
                    result = self.dm.download_whisper_model(self.WhisperModelType.TINY)
        self.assertTrue(result)
        self.assertFalse(called)

    def test_download_model_insufficient_space(self):
        called = False

        def fake_download(*args, **kwargs):
            nonlocal called
            called = True

        with unittest.mock.patch.object(self.dm, 'download_model', side_effect=fake_download):
            with unittest.mock.patch.object(Path, 'exists', return_value=False):
                with unittest.mock.patch.object(self.dm, 'check_disk_space', return_value=False):
                    result = self.dm.download_whisper_model(self.WhisperModelType.TINY)
        self.assertFalse(result)
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
