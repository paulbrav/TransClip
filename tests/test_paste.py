import unittest
from unittest.mock import patch

from granite_speach.paste import SystemPasteInjector, detect_clipboard_backend, paste_transcript
from granite_speach.platform_capabilities import clipboard_capability, session_info
from granite_speach.settings import Settings


class FakeClipboard:
    backend_name = "fake-clipboard"

    def __init__(self, initial: str):
        self.value = initial

    def read(self) -> str:
        return self.value

    def write(self, text: str) -> None:
        self.value = text


class FakeInjector:
    def __init__(self, pasted: bool = True):
        self.pasted = pasted
        self.backend_name = "fake-paste" if pasted else None

    def paste(self) -> bool:
        return self.pasted


class PasteTests(unittest.TestCase):
    def test_restores_clipboard_when_safe(self):
        settings = Settings(clipboard_restore_delay_ms=0, restore_clipboard_after_paste=True)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, FakeInjector())

        self.assertTrue(result.copied)
        self.assertTrue(result.pasted)
        self.assertTrue(result.restored)
        self.assertEqual(clipboard.value, "prior")
        self.assertEqual(result.clipboard_backend, "fake-clipboard")
        self.assertEqual(result.paste_backend, "fake-paste")

    def test_leaves_transcript_when_paste_fails(self):
        settings = Settings(clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, FakeInjector(False))

        self.assertFalse(result.pasted)
        self.assertFalse(result.restored)
        self.assertEqual(clipboard.value, "transcript")

    def test_missing_system_clipboard_is_structured_failure(self):
        with patch("granite_speach.paste.SystemClipboard", side_effect=RuntimeError("missing clipboard")):
            result = paste_transcript("transcript", Settings())

        self.assertFalse(result.copied)
        self.assertFalse(result.pasted)
        self.assertEqual(result.clipboard_backend, "unavailable")
        self.assertIn("missing clipboard", result.error_detail)

    def test_unreadable_prior_clipboard_still_copies_transcript(self):
        class UnreadablePriorClipboard(FakeClipboard):
            def __init__(self):
                super().__init__("prior")
                self.reads = 0

            def read(self):
                self.reads += 1
                if self.reads == 1:
                    raise UnicodeDecodeError("utf-8", b"\x89", 0, 1, "invalid")
                return self.value

        clipboard = UnreadablePriorClipboard()
        result = paste_transcript(
            "transcript",
            Settings(restore_clipboard_after_paste=True, clipboard_restore_delay_ms=0),
            clipboard,
            FakeInjector(),
        )

        self.assertTrue(result.copied)
        self.assertTrue(result.pasted)
        self.assertFalse(result.restored)
        self.assertTrue(result.transcript_left_on_clipboard)
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

    def test_injector_exception_is_structured_failure(self):
        class RaisingInjector:
            def paste(self):
                raise RuntimeError("paste exploded")

        settings = Settings(clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, RaisingInjector())

        self.assertTrue(result.copied)
        self.assertFalse(result.pasted)
        self.assertIn("paste exploded", result.error_detail)

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
            patch.dict("granite_speach.paste.os.environ", {"XDG_SESSION_TYPE": "x11"}),
            patch("granite_speach.paste.shutil.which", side_effect=which),
            patch("granite_speach.paste.subprocess.run", side_effect=run),
        ):
            injector = SystemPasteInjector()
            self.assertTrue(injector.paste())

        self.assertEqual(calls, ["xdotool"])
        self.assertEqual(injector.backend_name, "xdotool")

    def test_wayland_system_injector_tries_ydotool_after_wtype_failure(self):
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
            patch.dict("granite_speach.paste.os.environ", {"XDG_SESSION_TYPE": "wayland"}),
            patch("granite_speach.paste.shutil.which", side_effect=which),
            patch("granite_speach.paste.subprocess.run", side_effect=run),
        ):
            injector = SystemPasteInjector()
            self.assertTrue(injector.paste())

        self.assertEqual(calls, ["wtype", "ydotool"])
        self.assertEqual(ydotool_command, ["ydotool", "key", "ctrl+v"])
        self.assertEqual(injector.backend_name, "ydotool")

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

    def test_wayland_clipboard_requires_wl_clipboard(self):
        with (
            patch("granite_speach.paste.platform.system", return_value="Linux"),
            patch.dict("granite_speach.paste.os.environ", {"XDG_SESSION_TYPE": "wayland"}),
            patch("granite_speach.paste.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "Wayland clipboard requires wl-clipboard"),
        ):
            detect_clipboard_backend()

    def test_x11_clipboard_can_use_xclip(self):
        def which(name):
            return f"/usr/bin/{name}" if name == "xclip" else None

        with (
            patch("granite_speach.paste.platform.system", return_value="Linux"),
            patch.dict("granite_speach.paste.os.environ", {"XDG_SESSION_TYPE": "x11"}),
            patch("granite_speach.paste.shutil.which", side_effect=which),
        ):
            backend = detect_clipboard_backend()

        self.assertEqual(backend.name, "xclip")

    def test_wayland_display_indicator_prefers_wl_clipboard_when_session_unknown(self):
        def which(name):
            return f"/usr/bin/{name}" if name in {"xclip", "wl-copy", "wl-paste"} else None

        info = session_info(
            environ={"WAYLAND_DISPLAY": "wayland-0"},
            system="Linux",
        )
        capability = clipboard_capability(which=which, info=info)

        self.assertTrue(capability.ok)
        self.assertEqual(capability.backend, "wl-clipboard")


if __name__ == "__main__":
    unittest.main()
