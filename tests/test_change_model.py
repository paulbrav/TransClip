import importlib
import sys
import types
import unittest
from typing import Any, cast
from unittest import mock


class DummyTray:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, object, int]] = []

    def showMessage(self, title: str, text: str, icon: object, timeout: int) -> None:
        self.messages.append((title, text, icon, timeout))


def create_app(app_mod: types.ModuleType) -> Any:
    TransClip = app_mod.TransClip
    WhisperModelType = app_mod.WhisperModelType
    app = TransClip.__new__(TransClip)
    app.recording = False
    app.processing = False
    app.tray = DummyTray()
    app.has_cuda = lambda: False
    app.model = object()
    app.current_model_type = WhisperModelType.BASE
    return app


class ChangeModelTests(unittest.TestCase):
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
            {
                "TINY": "tiny",
                "BASE": "base",
                "get_description": classmethod(lambda cls, m: str(m)),
            },
        )
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
        qcore.QTimer = object
        modules["numpy"].float32 = float
        modules["scipy.signal"].resample = lambda data, samples: data
        keyboard_mod = modules["pynput.keyboard"]
        keyboard_mod.Key = type("Key", (), {"home": "home"})
        keyboard_mod.KeyCode = object
        keyboard_mod.Listener = object
        keyboard_mod.Controller = object
        modules["pynput"].keyboard = keyboard_mod
        self.patch = mock.patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        import transclip.app as app_mod
        importlib.reload(app_mod)
        self.app_mod = app_mod

    def test_change_model_updates_state(self) -> None:
        WhisperModelType = self.app_mod.WhisperModelType
        app = create_app(self.app_mod)
        with mock.patch.object(self.app_mod, "WhisperModel", return_value="new_model"):
            app.change_model(WhisperModelType.TINY)
        self.assertEqual(app.model, "new_model")
        self.assertEqual(app.current_model_type, WhisperModelType.TINY)
        tray = cast(DummyTray, app.tray)
        self.assertTrue(any("Changed model" in msg[1] for msg in tray.messages))


if __name__ == "__main__":
    unittest.main()
