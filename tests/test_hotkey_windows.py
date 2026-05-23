import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from transclip.hotkey_windows import install_hotkey, start_windows_hotkey
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime


class HotkeyWindowsTests(unittest.TestCase):
    def test_install_hotkey_is_tray_owned_noop(self):
        ok, detail = install_hotkey()

        self.assertTrue(ok)
        self.assertIn("tray", detail.lower())

    def test_start_windows_hotkey_registers_binding_from_settings(self):
        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))
        settings = Settings(hotkey_windows="ctrl+alt+t")
        callback_calls = []

        def callback() -> None:
            callback_calls.append(True)

        keyboard = types.ModuleType("keyboard")
        keyboard.add_hotkey = MagicMock(return_value="handle-1")
        keyboard.remove_hotkey = MagicMock()

        with patch.dict(sys.modules, {"keyboard": keyboard}):
            stop = start_windows_hotkey(callback, settings, runtime)
            stop()

        keyboard.add_hotkey.assert_called_once_with("ctrl+alt+t", callback, suppress=False)
        keyboard.remove_hotkey.assert_called_once_with("handle-1")

    def test_start_windows_hotkey_rejects_non_windows_runtime(self):
        runtime = FakeRuntime(system="Linux", home=Path("/home/test"))
        with self.assertRaisesRegex(RuntimeError, "only available on Windows"):
            start_windows_hotkey(lambda: None, Settings(), runtime)


if __name__ == "__main__":
    unittest.main()
