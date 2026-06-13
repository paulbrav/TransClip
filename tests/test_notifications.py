import sys
import unittest
from unittest.mock import patch


class ToastXmlTests(unittest.TestCase):
    def test_toast_xml_escapes_and_includes_text(self) -> None:
        from transclip.desktop.notifications import toast_xml

        xml = toast_xml("Title & <b>", "Body line")
        self.assertTrue(xml.startswith("<toast>"))
        self.assertIn("Title &amp; &lt;b&gt;", xml)
        self.assertIn("<text>Body line</text>", xml)


class WindowsToastTests(unittest.TestCase):
    def test_returns_false_when_winrt_unavailable(self) -> None:
        from transclip.desktop import notifications

        with patch.object(notifications, "_winrt_toast_api", side_effect=ImportError):
            self.assertFalse(notifications.windows_toast("t", "m"))

    def test_drives_winrt_with_app_id_and_text(self) -> None:
        from transclip.desktop import notifications

        calls: dict[str, object] = {}

        class FakeDoc:
            def load_xml(self, xml: str) -> None:
                calls["xml"] = xml

        class FakeToast:
            def __init__(self, doc: object) -> None:
                calls["doc"] = doc

        class FakeManager:
            @staticmethod
            def create_toast_notifier_with_id(app_id: str):
                calls["app_id"] = app_id

                class _Notifier:
                    def show(self, toast: object) -> None:
                        calls["shown"] = toast

                return _Notifier()

        with patch.object(notifications, "_winrt_toast_api", return_value=(FakeDoc, FakeToast, FakeManager)):
            ok = notifications.windows_toast("Hello", "World", app_id="com.test.App")

        self.assertTrue(ok)
        self.assertEqual(calls["app_id"], "com.test.App")
        self.assertIn("Hello", calls["xml"])
        self.assertIn("World", calls["xml"])
        self.assertIsInstance(calls["shown"], FakeToast)

    def test_returns_false_if_show_raises(self) -> None:
        from transclip.desktop import notifications

        class Boom:
            @staticmethod
            def create_toast_notifier_with_id(app_id: str):
                raise RuntimeError("no shell")

        with patch.object(notifications, "_winrt_toast_api", return_value=(object, object, Boom)):
            self.assertFalse(notifications.windows_toast("t", "m"))


@unittest.skipUnless(sys.platform == "win32", "writes to HKCU")
class RegisterToastAppIdLiveTests(unittest.TestCase):
    def test_register_writes_display_name(self) -> None:
        import winreg

        from transclip.desktop.notifications import register_toast_app_id

        app_id = "com.paulbrav.TransClip.RegTest"
        key_path = rf"Software\Classes\AppUserModelId\{app_id}"
        self.assertTrue(register_toast_app_id(app_id, "TransClip Reg Test"))
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "DisplayName")
            self.assertEqual(value, "TransClip Reg Test")
        finally:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)


if __name__ == "__main__":
    unittest.main()
