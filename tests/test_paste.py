import unittest
from unittest.mock import patch

from granite_speach.paste import SystemPasteInjector, paste_transcript
from granite_speach.settings import Settings


class FakeClipboard:
    def __init__(self, initial: str):
        self.value = initial

    def read(self) -> str:
        return self.value

    def write(self, text: str) -> None:
        self.value = text


class FakeInjector:
    def __init__(self, pasted: bool = True):
        self.pasted = pasted

    def paste(self) -> bool:
        return self.pasted


class PasteTests(unittest.TestCase):
    def test_restores_clipboard_when_safe(self):
        settings = Settings(clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, FakeInjector())

        self.assertTrue(result.pasted)
        self.assertTrue(result.restored)
        self.assertEqual(clipboard.value, "prior")

    def test_leaves_transcript_when_paste_fails(self):
        settings = Settings(clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, FakeInjector(False))

        self.assertFalse(result.pasted)
        self.assertFalse(result.restored)
        self.assertEqual(clipboard.value, "transcript")

    def test_injector_error_detail_is_reported(self):
        class ErrorInjector:
            def paste(self):
                return False

            def error_detail(self):
                return "wtype paste command failed: virtual keyboard unsupported"

        settings = Settings(clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, ErrorInjector())

        self.assertFalse(result.pasted)
        self.assertIn("virtual keyboard", result.error_detail)

    def test_system_injector_tries_xdotool_after_wtype_failure(self):
        calls = []

        def which(name):
            return f"/usr/bin/{name}" if name in {"wtype", "xdotool"} else None

        def run(command, **_kwargs):
            calls.append(command[0])
            if command[0] == "wtype":
                return type("Completed", (), {"returncode": 1, "stdout": "virtual keyboard unsupported"})()
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        with (
            patch("granite_speach.paste.platform.system", return_value="Linux"),
            patch("granite_speach.paste.shutil.which", side_effect=which),
            patch("granite_speach.paste.subprocess.run", side_effect=run),
        ):
            injector = SystemPasteInjector()
            self.assertTrue(injector.paste())

        self.assertEqual(calls, ["wtype", "xdotool"])

    def test_system_injector_tries_ydotool_after_wtype_and_xdotool_failure(self):
        calls = []

        def which(name):
            return f"/usr/bin/{name}" if name in {"wtype", "xdotool", "ydotool"} else None

        ydotool_command = []

        def run(command, **_kwargs):
            calls.append(command[0])
            if command[0] in {"wtype", "xdotool"}:
                return type("Completed", (), {"returncode": 1, "stdout": f"{command[0]} failed"})()
            ydotool_command.extend(command)
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        with (
            patch("granite_speach.paste.platform.system", return_value="Linux"),
            patch("granite_speach.paste.shutil.which", side_effect=which),
            patch("granite_speach.paste.subprocess.run", side_effect=run),
        ):
            injector = SystemPasteInjector()
            self.assertTrue(injector.paste())

        self.assertEqual(calls, ["wtype", "xdotool", "ydotool"])
        self.assertEqual(ydotool_command, ["ydotool", "key", "ctrl+v"])

    def test_system_injector_reports_runtime_failure(self):
        def which(name):
            return f"/usr/bin/{name}" if name == "wtype" else None

        def run(_command, **_kwargs):
            return type("Completed", (), {"returncode": 1, "stdout": "virtual keyboard unsupported"})()

        with (
            patch("granite_speach.paste.platform.system", return_value="Linux"),
            patch("granite_speach.paste.shutil.which", side_effect=which),
            patch("granite_speach.paste.subprocess.run", side_effect=run),
        ):
            injector = SystemPasteInjector()
            self.assertFalse(injector.paste())
            self.assertIn("virtual keyboard unsupported", injector.error_detail())


if __name__ == "__main__":
    unittest.main()
