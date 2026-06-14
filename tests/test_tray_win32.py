import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _has_pystray() -> bool:
    # PystrayMenuSinkActionTests imports pystray directly; without the windows-ui
    # extra it would error rather than skip. The other win32 tray classes import
    # win32 lazily and do not need it.
    return importlib.util.find_spec("pystray") is not None


@unittest.skipUnless(sys.platform == "win32", "tray win32 adapter requires pystray/Pillow")
class TrayWin32HotkeyTests(unittest.TestCase):
    def test_invalid_binding_is_rejected_without_persisting_or_restarting(self) -> None:
        from transclip.desktop.tray import win32

        session = MagicMock()
        session.settings.hotkey_windows = "ctrl+shift+space"
        restart = MagicMock()
        with (
            patch.object(win32, "is_valid_hotkey", return_value=False),
            patch.object(win32, "patch_settings") as patch_settings_mock,
        ):
            win32._apply_hotkey_selection(session, "definitely not!!", restart)

        patch_settings_mock.assert_not_called()
        restart.assert_not_called()

    def test_blank_binding_leaves_hotkey_unchanged(self) -> None:
        from transclip.desktop.tray import win32

        session = MagicMock()
        restart = MagicMock()
        with patch.object(win32, "patch_settings") as patch_settings_mock:
            win32._apply_hotkey_selection(session, "   ", restart)

        patch_settings_mock.assert_not_called()
        restart.assert_not_called()

    def test_valid_binding_is_persisted_and_registered(self) -> None:
        from transclip.desktop.tray import win32

        session = MagicMock()
        session.explicit_settings_path = None
        restart = MagicMock()
        with (
            patch.object(win32, "is_valid_hotkey", return_value=True),
            patch.object(win32, "patch_settings") as patch_settings_mock,
            patch.object(win32, "settings_path", return_value="cfg"),
            patch.object(win32, "windows_toast") as toast_mock,
        ):
            win32._apply_hotkey_selection(session, "ctrl+alt+t", restart)

        patch_settings_mock.assert_called_once()
        self.assertEqual(patch_settings_mock.call_args.kwargs["hotkey_windows"], "ctrl+alt+t")
        restart.assert_called_once()
        # The new binding is confirmed with a toast.
        toast_mock.assert_called_once()
        self.assertIn("ctrl+alt+t", toast_mock.call_args.args[1])


@unittest.skipUnless(sys.platform == "win32", "tray win32 adapter requires pystray/Pillow")
class TrayWin32RecordingNotifyTests(unittest.TestCase):
    def _fake_outcome(self, *, ok=True, action="started", paste_failed="", error=""):
        return SimpleNamespace(
            ok=ok, payload={"action": action}, paste_failed_message=paste_failed, error_message=error
        )

    def test_notifies_on_start_when_enabled(self) -> None:
        from transclip.desktop.tray import win32

        with patch.object(win32, "windows_toast") as toast:
            win32._notify_toggle(self._fake_outcome(action="started"), SimpleNamespace(tray_notifications=True))

        toast.assert_called_once()
        self.assertIn("Recording", toast.call_args.args[1])

    def test_suppressed_when_setting_disabled(self) -> None:
        from transclip.desktop.tray import win32

        with patch.object(win32, "windows_toast") as toast:
            win32._notify_toggle(self._fake_outcome(action="started"), SimpleNamespace(tray_notifications=False))

        toast.assert_not_called()

    def test_blocked_paste_is_surfaced(self) -> None:
        from transclip.desktop.tray import win32

        with patch.object(win32, "windows_toast") as toast:
            win32._notify_toggle(
                self._fake_outcome(action="stopped", paste_failed="UIPI blocked"),
                SimpleNamespace(tray_notifications=True),
            )

        self.assertIn("Ctrl+V", toast.call_args.args[1])


@unittest.skipUnless(sys.platform == "win32", "tray win32 adapter requires pystray/Pillow")
class TrayWin32ActionNotifyTests(unittest.TestCase):
    def test_notify_action_toasts_when_enabled(self) -> None:
        from transclip.desktop.tray import win32

        with patch.object(win32, "windows_toast") as toast:
            win32._notify_action("Restarting the dictation service", SimpleNamespace(tray_notifications=True))

        toast.assert_called_once()
        self.assertIn("Restarting the dictation service", toast.call_args.args[1])

    def test_notify_action_suppressed_when_disabled(self) -> None:
        from transclip.desktop.tray import win32

        with patch.object(win32, "windows_toast") as toast:
            win32._notify_action("anything", SimpleNamespace(tray_notifications=False))

        toast.assert_not_called()


class CaptureHotkeyTests(unittest.TestCase):
    def test_pauses_reads_then_resumes_in_order(self) -> None:
        from transclip.desktop.tray.win32 import _capture_hotkey

        calls: list[str] = []

        def read() -> str:
            calls.append("read")
            return "ctrl+shift+space"

        result = _capture_hotkey(
            pause=lambda: calls.append("pause"),
            resume=lambda: calls.append("resume"),
            read_hotkey=read,
        )

        self.assertEqual(result, "ctrl+shift+space")
        self.assertEqual(calls, ["pause", "read", "resume"])

    def test_always_resumes_and_returns_none_when_read_fails(self) -> None:
        from transclip.desktop.tray.win32 import _capture_hotkey

        calls: list[str] = []

        def boom() -> str:
            calls.append("read")
            raise RuntimeError("hook failed")

        result = _capture_hotkey(
            pause=lambda: calls.append("pause"),
            resume=lambda: calls.append("resume"),
            read_hotkey=boom,
        )

        self.assertIsNone(result)
        self.assertEqual(calls, ["pause", "read", "resume"])  # resume runs even on failure


@unittest.skipUnless(sys.platform == "win32", "tray win32 adapter requires pystray/Pillow")
class TrayWin32IconTests(unittest.TestCase):
    def test_health_icon_color_maps_known_states(self) -> None:
        from transclip.desktop.tray.win32 import health_icon_color

        self.assertEqual(health_icon_color("recording"), "red")
        self.assertEqual(health_icon_color("ready"), "green")
        self.assertEqual(health_icon_color("offline"), "orange")

    def test_health_icon_color_falls_back_for_unknown_state(self) -> None:
        from transclip.desktop.tray.win32 import health_icon_color

        # A new health state must not raise KeyError: build_image runs inside
        # the pystray event loop, so an unmapped name would crash the tray.
        self.assertEqual(health_icon_color("busy"), "gray")


@unittest.skipUnless(sys.platform == "win32" and _has_pystray(), "real pystray validates menu actions")
class PystrayMenuSinkActionTests(unittest.TestCase):
    """The real pystray rejects menu actions taking >2 positional args.

    The other tray tests inject a fake pystray that does not enforce this, so a
    closure with a default-arg binding (which inflates co_argcount) passes them
    but crashes the live tray. These build the menus with the real pystray.
    """

    def _sink(self):
        import pystray
        from transclip.desktop.tray.sinks.win32 import PystrayMenuSink

        items: list = []
        refs: dict = {}
        sink = PystrayMenuSink(
            items,
            refs,
            pystray=pystray,
            after_action=lambda fn: fn(),
            set_model=lambda model_id, backend: None,
            on_copy_history=lambda value: None,
        )
        return sink, items, refs

    def test_model_submenu_actions_are_accepted_by_pystray(self) -> None:
        sink, items, _ = self._sink()
        choices = [("Granite AR", SimpleNamespace(model_id="ibm-granite/x", backend="granite"))]

        sink.model_submenu("models", "ASR model", choices)  # raises ValueError pre-fix

        self.assertEqual(len(items), 1)

    def test_history_submenu_actions_are_accepted_by_pystray(self) -> None:
        sink, _, refs = self._sink()
        refs["_history_entries"] = [("preview", "full transcript text")]

        menu = sink._build_history_menu()  # raises ValueError pre-fix

        self.assertIsNotNone(menu)

    def test_status_label_is_updatable_after_creation(self) -> None:
        # pystray MenuItem.text is read-only; the menu refresh updates the
        # backing state and the item must re-read it.
        sink, items, refs = self._sink()
        sink.status_label("status_item", "Service: starting")

        self.assertEqual(items[0].text, "Service: starting")
        refs["status_item"].text = "Service: ready"  # what the view's setter does
        self.assertEqual(items[0].text, "Service: ready")

    def test_action_item_label_and_enabled_are_updatable(self) -> None:
        sink, items, refs = self._sink()
        sink.action("toggle_item", "Record", None, enabled=True, callback=lambda *_: None)

        self.assertEqual(items[0].text, "Record")
        self.assertTrue(items[0].enabled)
        refs["toggle_item"].text = "Stop + paste"
        refs["toggle_item"].enabled = False
        self.assertEqual(items[0].text, "Stop + paste")
        self.assertFalse(items[0].enabled)


if __name__ == "__main__":
    unittest.main()
