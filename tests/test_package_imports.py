import unittest


class PackageImportTests(unittest.TestCase):
    def test_public_entry_points_import_cleanly(self):
        from transclip.daemon import (
            CommandResult,
            ServiceState,
            append_toggle_log,
            collect_status,
            install_daemon,
            toggle_log_path,
        )
        from transclip.desktop.hotkey import (
            build_toggle_command,
            build_toggle_invocation,
            get_gnome_shortcut_status,
            hotkey_setup_message,
            install_shortcut,
            shortcut_readiness,
            start_windows_hotkey,
        )
        from transclip.desktop.paste import SystemClipboard, paste_capability
        from transclip.desktop.tray import run_tray
        from transclip.doctor import Check, run_checks
        from transclip.platform.runtime import PlatformRuntime, get_runtime, open_path

        self.assertTrue(callable(install_daemon))
        self.assertTrue(callable(install_shortcut))
        self.assertTrue(callable(get_gnome_shortcut_status))
        self.assertTrue(callable(shortcut_readiness))
        self.assertTrue(callable(run_tray))
        self.assertTrue(callable(run_checks))
        self.assertTrue(callable(build_toggle_command))
        self.assertTrue(callable(hotkey_setup_message))
        self.assertTrue(callable(start_windows_hotkey))
        self.assertTrue(callable(get_runtime))
        self.assertTrue(callable(collect_status))
        self.assertTrue(callable(append_toggle_log))
        self.assertIsNotNone(Check)
        self.assertIsNotNone(CommandResult)
        self.assertIsNotNone(ServiceState)
        self.assertIsNotNone(PlatformRuntime)
        self.assertIsNotNone(SystemClipboard)
        self.assertIsNotNone(paste_capability)
        self.assertIsNotNone(toggle_log_path)
        self.assertIsNotNone(build_toggle_invocation)
        self.assertIsNotNone(open_path)


if __name__ == "__main__":
    unittest.main()
