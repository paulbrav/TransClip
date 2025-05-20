import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class DummyTray:
    def showMessage(self, title: str, text: str, icon: object, timeout: int) -> None:
        pass


def create_app(app_mod: types.ModuleType) -> object:
    TransClip = app_mod.TransClip
    app = TransClip.__new__(TransClip)
    app.tray = DummyTray()
    app.cleaner = app_mod.Cleaner({"cleanup_prompt": "OLD {text}"})
    return app


class CleanupPromptMenuTests(unittest.TestCase):
    def setUp(self) -> None:
        modules = {
            "numpy": types.ModuleType("numpy"),
            "sounddevice": types.ModuleType("sounddevice"),
            "PyQt5": types.ModuleType("PyQt5"),
            "PyQt5.QtCore": types.ModuleType("QtCore"),
            "PyQt5.QtGui": types.ModuleType("QtGui"),
            "PyQt5.QtWidgets": types.ModuleType("QtWidgets"),
            "scipy": types.ModuleType("scipy"),
            "scipy.signal": types.ModuleType("signal"),
            "faster_whisper": types.ModuleType("faster_whisper"),
            "transclip.transcription": types.ModuleType("transclip.transcription"),
            "pynput": types.ModuleType("pynput"),
            "pynput.keyboard": types.ModuleType("pynput.keyboard"),
            "pyperclip": types.ModuleType("pyperclip"),
            "transclip.cleanup": types.ModuleType("transclip.cleanup"),
        }
        modules["transclip.transcription"].WhisperModelType = type(
            "WhisperModelType",
            (),
            {"BASE": "base", "get_description": classmethod(lambda cls, m: str(m))},
        )
        modules["transclip.transcription"].get_model_path = lambda m: str(m)
        modules["transclip.transcription"].DEFAULT_MODEL_TYPE = (
            modules["transclip.transcription"].WhisperModelType.BASE
        )
        modules["transclip.transcription"].TranscriptionWorker = object
        modules["transclip.transcription"].NeMoTranscriptionWorker = object
        modules["transclip.transcription"].load_nemo_model = lambda m: "nemo_model"
        modules["faster_whisper"].WhisperModel = object
        qtwidgets = modules["PyQt5.QtWidgets"]
        qtwidgets.QSystemTrayIcon = type(
            "QSystemTrayIcon",
            (),
            {"Information": 0, "Warning": 1, "Critical": 2},
        )
        qtwidgets.QActionGroup = object
        qtwidgets.QApplication = object
        qtwidgets.QDialog = object
        qtwidgets.QHBoxLayout = object
        qtwidgets.QLabel = object
        qtwidgets.QMenu = object
        qtwidgets.QPushButton = object
        qtwidgets.QVBoxLayout = object
        qtwidgets.QInputDialog = type(
            "QInputDialog", (), {"getMultiLineText": staticmethod(lambda *a, **k: ("", False))}
        )
        modules["PyQt5.QtGui"].QIcon = object
        qcore = modules["PyQt5.QtCore"]
        qcore.QObject = object
        qcore.QThread = object
        qcore.QTimer = object
        modules["numpy"].float32 = float
        modules["scipy.signal"].resample = lambda data, samples: data
        keyboard_mod = modules["pynput.keyboard"]
        keyboard_mod.Key = type("KeyEnum", (), {"home": "home"})
        keyboard_mod.KeyCode = object
        keyboard_mod.Listener = object
        keyboard_mod.Controller = object
        modules["pynput"].keyboard = keyboard_mod

        class DummyCleaner:
            def __init__(self, cfg: dict) -> None:
                self.prompt_template = cfg.get("cleanup_prompt", "")

            def __call__(self, text: str) -> str:
                return text

        modules["transclip.cleanup"].Cleaner = DummyCleaner
        self.patch = mock.patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)

        import transclip.config as config_mod
        importlib.reload(config_mod)
        self.config_mod = config_mod

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_dir = Path(self.temp_dir.name) / "transclip"
        self.config_file = self.config_dir / "config.json"
        patch1 = mock.patch.object(config_mod, "CONFIG_DIR", self.config_dir)
        patch2 = mock.patch.object(config_mod, "CONFIG_FILE", self.config_file)
        patch1.start()
        patch2.start()
        self.addCleanup(patch1.stop)
        self.addCleanup(patch2.stop)

        import transclip.app as app_mod
        importlib.reload(app_mod)
        self.app_mod = app_mod

    def test_prompt_saved(self) -> None:
        app = create_app(self.app_mod)
        with mock.patch.object(
            self.app_mod.QInputDialog,
            "getMultiLineText",
            return_value=("NEW {text}", True),
        ):
            app.show_cleanup_prompt_dialog()
        config = self.config_mod.load_config()
        self.assertEqual(config["cleanup_prompt"], "NEW {text}")


if __name__ == "__main__":
    unittest.main()
