import ctypes
import sys
import unittest
from unittest.mock import MagicMock, patch

from transclip.desktop.paste.win32 import (
    read_clipboard_text,
    send_ctrl_v_paste,
    write_clipboard_text,
)


class Win32ClipboardTests(unittest.TestCase):
    @patch("transclip.desktop.paste.win32.platform.system", return_value="Linux")
    def test_read_clipboard_rejects_non_windows(self, _system):
        with self.assertRaisesRegex(RuntimeError, "only available on Windows"):
            read_clipboard_text()

    @patch("transclip.desktop.paste.win32.platform.system", return_value="Linux")
    def test_write_clipboard_rejects_non_windows(self, _system):
        with self.assertRaisesRegex(RuntimeError, "only available on Windows"):
            write_clipboard_text("hello")

    @patch("transclip.desktop.paste.win32.platform.system", return_value="Linux")
    def test_send_ctrl_v_paste_rejects_non_windows(self, _system):
        with self.assertRaisesRegex(RuntimeError, "only available on Windows"):
            send_ctrl_v_paste()

    @patch("transclip.desktop.paste.win32.platform.system", return_value="Windows")
    @patch("transclip.desktop.paste.win32.time.sleep")
    def test_read_clipboard_text_returns_unicode_payload(self, _sleep, _system):
        user32 = MagicMock()
        kernel32 = MagicMock()
        user32.OpenClipboard.return_value = True
        user32.GetClipboardData.return_value = 42
        kernel32.GlobalLock.return_value = ctypes.c_wchar_p("hello clipboard").value

        with patch.object(ctypes, "windll", MagicMock(user32=user32, kernel32=kernel32), create=True):
            text = read_clipboard_text()

        self.assertEqual(text, "hello clipboard")
        user32.CloseClipboard.assert_called_once()

    @patch("transclip.desktop.paste.win32.platform.system", return_value="Windows")
    def test_write_clipboard_text_sets_unicode_payload(self, _system):
        user32 = MagicMock()
        kernel32 = MagicMock()
        user32.OpenClipboard.return_value = True
        user32.EmptyClipboard.return_value = True
        user32.SetClipboardData.return_value = True
        kernel32.GlobalAlloc.return_value = 99
        kernel32.GlobalLock.return_value = ctypes.create_string_buffer(64)

        with patch.object(ctypes, "windll", MagicMock(user32=user32, kernel32=kernel32), create=True):
            write_clipboard_text("saved")

        user32.SetClipboardData.assert_called_once()
        user32.CloseClipboard.assert_called_once()

    @patch("transclip.desktop.paste.win32.platform.system", return_value="Windows")
    @patch("transclip.desktop.paste.win32.time.sleep")
    def test_send_ctrl_v_paste_uses_sendinput(self, _sleep, _system):
        user32 = MagicMock()
        user32.SendInput.return_value = 4

        with patch.object(ctypes, "windll", MagicMock(user32=user32), create=True):
            send_ctrl_v_paste()

        user32.SendInput.assert_called_once()
        sent_count, _array, _size = user32.SendInput.call_args.args
        self.assertEqual(sent_count, 4)


@unittest.skipUnless(sys.platform == "win32", "exercises the real Win32 clipboard")
class Win32ClipboardLiveTests(unittest.TestCase):
    """Round-trip against the real OS clipboard.

    The mocked tests above replace ctypes wholesale, so they cannot catch a
    64-bit handle/pointer being truncated by a missing ``restype`` (the value
    faults on dereference). Driving the real Win32 calls is the only way to
    prove the clipboard actually works on a 64-bit Windows host.
    """

    def setUp(self) -> None:
        self._prior = read_clipboard_text()

    def tearDown(self) -> None:
        write_clipboard_text(self._prior)

    def test_unicode_round_trip(self) -> None:
        sample = "TransClip 12345 ✓ 日本語"
        write_clipboard_text(sample)
        self.assertEqual(read_clipboard_text(), sample)

    def test_empty_string_round_trip(self) -> None:
        write_clipboard_text("")
        self.assertEqual(read_clipboard_text(), "")


if __name__ == "__main__":
    unittest.main()
