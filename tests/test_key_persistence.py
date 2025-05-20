import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class DummyTray:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, object, int]] = []

    def showMessage(self, title: str, text: str, icon: object, timeout: int) -> None:
        self.messages.append((title, text, icon, timeout))


def create_app(app_mod: types.ModuleType) -> object:
    TransClip = app_mod.TransClip
    app = TransClip.__new__(TransClip)
    app.recording = False
    app.processing = False
    app.tray = DummyTray()
    app.has_cuda = lambda: False
    app.model = object()
    app.current_model_type = app_mod.WhisperModelType.BASE
    app.recording_key = app_mod.keyboard.Key.home
    return app


class KeyPersistenceTests(unittest.TestCase):
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
        modules["faster_whisper"].WhisperModel = object
        qtwidgets = modules["PyQt5.QtWidgets"]
        qtwidgets.QSystemTrayIcon = type(
            "QSystemTrayIcon",
            (),
            {"Information": 0, "Warning": 1, "Critical": 2},
        )
        for name in [
            "QActionGroup",
            "QApplication",
            "QDialog",
            "QHBoxLayout",
            "QLabel",
            "QMenu",
            "QPushButton",
            "QVBoxLayout",
        ]:
            setattr(qtwidgets, name, object)
        modules["PyQt5.QtGui"].QIcon = object
        qcore = modules["PyQt5.QtCore"]
        qcore.QObject = object
        qcore.QThread = object

        class QTimer:
            @staticmethod
            def singleShot(_ms: int, func):
                func()

        qcore.QTimer = QTimer
        modules["numpy"].float32 = float
        modules["scipy.signal"].resample = lambda data, samples: data
        keyboard_mod = modules["pynput.keyboard"]

        class DummyKey:
            def __init__(self, name: str) -> None:
                self.name = name

            def __str__(self) -> str:  # pragma: no cover - simple helper
                return f"Key.{self.name}"

            def __repr__(self) -> str:  # pragma: no cover
                return str(self)

        class DummyKeyCode:
            def __init__(self, char: str) -> None:
                self.char = char

            @classmethod
            def from_char(cls, char: str) -> "DummyKeyCode":
                return cls(char)

            def __str__(self) -> str:  # pragma: no cover
                return self.char

            def __repr__(self) -> str:  # pragma: no cover
                return f"'{self.char}'"

        keyboard_mod.Key = type("KeyEnum", (), {"home": DummyKey("home")})
        keyboard_mod.KeyCode = DummyKeyCode
        keyboard_mod.Listener = object
        keyboard_mod.Controller = object
        modules["pynput"].keyboard = keyboard_mod
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

    def test_change_key_persists(self) -> None:
        app = create_app(self.app_mod)
        key = self.app_mod.keyboard.Key.home
        app.change_recording_key(key)
        config = self.config_mod.load_config()
        self.assertEqual(config["recording_key"], "Key.home")


if __name__ == "__main__":
    unittest.main()
