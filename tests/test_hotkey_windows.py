import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from transclip.desktop.hotkey.windows import start_windows_hotkey
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime


class HotkeyWindowsTests(unittest.TestCase):
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

    def test_is_valid_hotkey_accepts_parseable_and_rejects_garbage(self):
        keyboard = types.ModuleType("keyboard")

        def parse_hotkey(binding):
            if not binding or "!" in binding:
                raise ValueError(f"unparseable hotkey {binding!r}")
            return [[(29, "ctrl")]]

        keyboard.parse_hotkey = parse_hotkey
        with patch.dict(sys.modules, {"keyboard": keyboard}):
            from transclip.desktop.hotkey.windows import is_valid_hotkey

            self.assertTrue(is_valid_hotkey("ctrl+shift+space"))
            self.assertFalse(is_valid_hotkey("definitely not a hotkey!!"))
            self.assertFalse(is_valid_hotkey(""))

    def test_stop_is_idempotent(self):
        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))
        removes = []

        def remove_hotkey(handle):
            removes.append(handle)
            if len(removes) > 1:
                raise KeyError(handle)  # keyboard raises if the hotkey is already gone

        keyboard = types.ModuleType("keyboard")
        keyboard.add_hotkey = MagicMock(return_value="handle-1")
        keyboard.remove_hotkey = remove_hotkey

        with patch.dict(sys.modules, {"keyboard": keyboard}):
            stop = start_windows_hotkey(lambda: None, Settings(), runtime)
            stop()
            stop()  # second stop must not raise

        self.assertEqual(removes, ["handle-1", "handle-1"])

    def test_pause_capture_resume_does_not_double_remove_hotkey(self):
        # Reproduces the Set-hotkey "Record keys" crash: pause stops the live
        # hotkey, then resume (restart) must not try to remove it again.
        from transclip.desktop.tray.win32 import _capture_hotkey

        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))
        registered: list[object] = []

        def add_hotkey(_binding, _cb, suppress=False):
            handle = object()
            registered.append(handle)
            return handle

        def remove_hotkey(handle):
            if handle not in registered:  # keyboard raises if already gone
                raise KeyError(handle)
            registered.remove(handle)

        keyboard = types.ModuleType("keyboard")
        keyboard.add_hotkey = add_hotkey
        keyboard.remove_hotkey = remove_hotkey

        with patch.dict(sys.modules, {"keyboard": keyboard}):
            holder: dict[str, object] = {"stop": None}

            def restart():
                stop = holder["stop"]
                if stop is not None:
                    stop()
                holder["stop"] = start_windows_hotkey(lambda: None, Settings(), runtime)

            def pause():
                stop = holder["stop"]
                if stop is not None:
                    stop()
                    holder["stop"] = None

            restart()  # initial registration
            result = _capture_hotkey(pause=pause, resume=restart, read_hotkey=lambda: "ctrl+alt+x")

        self.assertEqual(result, "ctrl+alt+x")
        self.assertEqual(len(registered), 1)  # exactly one hotkey live after capture

    def test_is_valid_hotkey_false_when_keyboard_missing(self):
        with patch.dict(sys.modules, {"keyboard": None}):
            from transclip.desktop.hotkey.windows import is_valid_hotkey

            self.assertFalse(is_valid_hotkey("ctrl+shift+space"))


if __name__ == "__main__":
    unittest.main()
