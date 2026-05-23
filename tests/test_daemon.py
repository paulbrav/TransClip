import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.daemon import (
    SERVICE_NAME,
    append_toggle_log,
    build_systemd_unit,
    collect_status,
    install_linux_daemon,
    install_macos_daemon,
    last_toggle_log_event,
    service_action,
    service_state,
    toggle_log_path,
)
from transclip.daemon.macos import launch_agent_path
from transclip.daemon.windows import build_task_scheduler_xml, install_windows_daemon
from transclip.desktop.hotkey import build_toggle_invocation, windows_hotkey_setup_message
from transclip.paths import service_settings_path
from transclip.product import TASK_SCHEDULER_NAME
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime, normalize_path_text


class DaemonTests(unittest.TestCase):
    def test_systemd_unit_uses_current_python_and_cli_serve(self):
        unit = build_systemd_unit(Path("/tmp/settings.toml"))

        self.assertIn("Description=TransClip dictation service", unit)
        settings_path = normalize_path_text(service_settings_path(Path("/tmp/settings.toml")))
        normalized_unit = normalize_path_text(unit).replace("'", "")
        self.assertIn(f"-m transclip.cli --settings {settings_path} serve", normalized_unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE", unit)

    def test_linux_install_writes_unit_runs_systemctl_and_installs_shortcut(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        shortcut = type(
            "Shortcut",
            (),
            {
                "binding": "<Super><Shift>XF86TouchpadOff",
                "command": "/bin/sh -lc transclip",
            },
        )()

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch(
                    "transclip.daemon.linux.install_shortcut",
                    return_value=shortcut,
                ) as install_shortcut,
            ):
                settings = Settings(hotkey_linux="<Control><Alt>space")
                results = install_linux_daemon(settings=settings, runner=runner, runtime=FakeRuntime(home=home))

            unit_path = home / ".config" / "systemd" / "user" / SERVICE_NAME
            self.assertTrue(unit_path.exists())
            self.assertIn("transclip.cli serve", unit_path.read_text(encoding="utf-8"))
            self.assertIn(["systemctl", "--user", "daemon-reload"], calls)
            self.assertIn(["systemctl", "--user", "enable", "--now", SERVICE_NAME], calls)
            self.assertEqual(install_shortcut.call_args.kwargs["binding"], "<Control><Alt>space")
            self.assertTrue(any("installed GNOME shortcut" in result.detail for result in results))
            self.assertTrue(all(result.ok for result in results))

    def test_toggle_log_is_jsonl_and_last_event_is_decoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toggle-record.log"
            append_toggle_log({"action": "started"}, path)
            append_toggle_log({"action": "stopped", "text": "hello"}, path)

            self.assertEqual(last_toggle_log_event(path), {"action": "stopped", "text": "hello"})

    def test_last_toggle_log_event_reads_trailing_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toggle-record.log"
            path.write_text('{"action": "started"}\n' + (" \n" * 10_000) + '{"action": "stopped"}', encoding="utf-8")

            self.assertEqual(last_toggle_log_event(path), {"action": "stopped"})

    def test_toggle_log_path_is_under_cache_dir_on_linux(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Linux", home=Path(tmp))
            self.assertEqual(
                toggle_log_path(runtime),
                Path(tmp) / ".cache" / "transclip" / "toggle-record.log",
            )

    def test_toggle_log_path_is_under_library_logs_on_darwin(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            self.assertEqual(
                toggle_log_path(runtime),
                Path(tmp) / "Library" / "Logs" / "transclip" / "toggle-record.log",
            )

    def test_toggle_log_path_is_under_localappdata_logs_on_windows(self):
        runtime = FakeRuntime(
            system="Windows",
            home=Path("C:/Users/tester"),
            env={"LOCALAPPDATA": "C:/Users/tester/AppData/Local"},
        )
        self.assertEqual(
            toggle_log_path(runtime),
            Path("C:/Users/tester/AppData/Local/transclip/logs/toggle-record.log"),
        )

    def test_macos_install_uses_launchctl_bootstrap(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            runtime = FakeRuntime(system="Darwin", home=home, check_output_text="501")
            results = install_macos_daemon(runner=runner, runtime=runtime)

            plist_path = home / "Library" / "LaunchAgents" / "com.paulbrav.transclip.plist"
            self.assertTrue(plist_path.exists())
            self.assertIn(["launchctl", "bootout", "gui/501/com.paulbrav.transclip"], calls)
            self.assertIn(["launchctl", "bootstrap", "gui/501", str(plist_path)], calls)
            self.assertTrue(any("Keyboard Shortcut" in result.detail for result in results))
            self.assertTrue(
                any(
                    "Library/Logs/transclip/toggle-record.log"
                    in normalize_path_text(result.detail)
                    for result in results
                )
            )
            self.assertTrue(all(result.ok for result in results))

    def test_macos_start_kickstarts_loaded_launch_agent(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return type("Completed", (), {"returncode": 0, "stdout": "state = running"})()

        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="501")
        result = service_action("start", runner=runner, runtime=runtime)

        self.assertTrue(result.ok)
        self.assertIn(["launchctl", "print", "gui/501/com.paulbrav.transclip"], calls)
        self.assertIn(["launchctl", "kickstart", "-k", "gui/501/com.paulbrav.transclip"], calls)
        self.assertNotIn(
            ["launchctl", "bootstrap", "gui/501", "/Users/test/Library/LaunchAgents/com.paulbrav.transclip.plist"],
            calls,
        )

    def test_macos_start_bootstraps_unloaded_launch_agent(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return_code = 1 if command[:2] == ["launchctl", "print"] else 0
            return type("Completed", (), {"returncode": return_code, "stdout": ""})()

        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="501")
        result = service_action("start", runner=runner, runtime=runtime)

        self.assertTrue(result.ok)
        self.assertIn(
            ["launchctl", "bootstrap", "gui/501", str(launch_agent_path(runtime))],
            calls,
        )

    def test_macos_restart_bootstraps_unloaded_launch_agent(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            if command[:2] == ["launchctl", "print"]:
                return type("Completed", (), {"returncode": 1, "stdout": "not loaded"})()
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="501")
        result = service_action("restart", runner=runner, runtime=runtime)

        self.assertTrue(result.ok)
        self.assertIn(
            ["launchctl", "bootstrap", "gui/501", str(launch_agent_path(runtime))],
            calls,
        )

    def test_macos_service_state_requires_running_launchd_job(self):
        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"), check_output_text="501")

        def loaded_not_running(_command, **_kwargs):
            return type("Completed", (), {"returncode": 0, "stdout": "state = exited"})()

        self.assertFalse(service_state(runner=loaded_not_running, runtime=runtime).active)

        def running(_command, **_kwargs):
            return type("Completed", (), {"returncode": 0, "stdout": "state = running\npid = 123"})()

        self.assertTrue(service_state(runner=running, runtime=runtime).active)

    def test_paste_status_reports_probe_failure(self):
        capability = type(
            "Capability",
            (),
            {"ok": False, "backend": None, "detail": "wtype unusable: unsupported"},
        )()
        clipboard_capability = type(
            "ClipboardCapability",
            (),
            {"ok": False, "backend": None, "detail": "No supported clipboard reader/writer found"},
        )()
        with (
            patch("transclip.daemon.status.paste_capability", return_value=capability),
            patch("transclip.daemon.status.clipboard_capability", return_value=clipboard_capability),
        ):
            status = collect_status(Settings(port=0), runtime=FakeRuntime(system="Other"))

        self.assertFalse(status["paste"]["ok"])
        self.assertIn("wtype unusable", status["paste"]["detail"])

    def test_windows_install_uses_schtasks_and_skips_gnome_shortcut(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            runtime = FakeRuntime(
                system="Windows",
                home=home,
                env={"LOCALAPPDATA": str(home / "AppData/Local")},
            )
            with patch("transclip.daemon.linux.install_shortcut") as install_shortcut:
                results = install_windows_daemon(
                    runner=runner,
                    runtime=runtime,
                    hotkey_setup_message=windows_hotkey_setup_message,
                )

            xml_path = home / "AppData/Local/transclip/logs/TransClip.xml"
            self.assertTrue(xml_path.exists())
            self.assertIn(["schtasks", "/Create", "/TN", TASK_SCHEDULER_NAME, "/XML", str(xml_path), "/F"], calls)
            self.assertIn(["schtasks", "/Run", "/TN", TASK_SCHEDULER_NAME], calls)
            install_shortcut.assert_not_called()
            self.assertTrue(any("ctrl+shift+space" in result.detail for result in results))
            self.assertTrue(any("prefetch" in result.detail for result in results))

    def test_windows_service_state_reports_running_task(self):
        runtime = FakeRuntime(
            system="Windows",
            home=Path("C:/Users/test"),
            env={"LOCALAPPDATA": "C:/Users/test/AppData/Local"},
        )

        def running(_command, **_kwargs):
            return type("Completed", (), {"returncode": 0, "stdout": "Status: Running"})()

        state = service_state(runner=running, runtime=runtime)
        self.assertTrue(state.active)

    def test_windows_service_action_runs_schtasks(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))
        result = service_action("start", runner=runner, runtime=runtime)

        self.assertTrue(result.ok)
        self.assertIn(["schtasks", "/Run", "/TN", TASK_SCHEDULER_NAME], calls)

    def test_task_scheduler_xml_quotes_settings_paths_with_spaces(self):
        settings_path = Path("C:/Users/test user/AppData/Roaming/transclip/settings.toml")
        xml = build_task_scheduler_xml(settings_path)

        self.assertIn('encoding="UTF-16"', xml)
        self.assertIn("<Enabled>true</Enabled>", xml)
        self.assertIn("<Hidden>true</Hidden>", xml)
        self.assertIn("test user", xml)
        self.assertIn("--settings", xml)
        self.assertIn("-m transclip.cli", xml)
        self.assertIn(" serve</Arguments>", xml)

    def test_service_settings_path_preserves_absolute_windows_paths(self):
        path = Path("C:/Users/test user/AppData/Roaming/transclip/settings.toml")
        self.assertEqual(
            normalize_path_text(service_settings_path(path)),
            "C:/Users/test user/AppData/Roaming/transclip/settings.toml",
        )

    def test_build_toggle_invocation_preserves_absolute_windows_settings_path(self):
        path = Path("C:/Users/test user/AppData/Roaming/transclip/settings.toml")
        command = build_toggle_invocation(path)
        self.assertIn(
            normalize_path_text(service_settings_path(path)),
            [normalize_path_text(str(part)) for part in command],
        )

    def test_windows_service_state_ignores_running_substring_without_status_line(self):
        from transclip.daemon.windows import _windows_task_reports_running

        self.assertFalse(_windows_task_reports_running("Last Result: Running tasks only"))
        self.assertTrue(_windows_task_reports_running("Status: Running"))


if __name__ == "__main__":
    unittest.main()
