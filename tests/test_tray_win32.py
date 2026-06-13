import sys
import unittest
from unittest.mock import MagicMock, patch


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
        ):
            win32._apply_hotkey_selection(session, "ctrl+alt+t", restart)

        patch_settings_mock.assert_called_once()
        self.assertEqual(patch_settings_mock.call_args.kwargs["hotkey_windows"], "ctrl+alt+t")
        restart.assert_called_once()


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


if __name__ == "__main__":
    unittest.main()
