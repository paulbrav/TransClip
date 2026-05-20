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
    last_toggle_log_event,
    toggle_log_path,
)
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime


class DaemonTests(unittest.TestCase):
    def test_systemd_unit_uses_current_python_and_cli_serve(self):
        unit = build_systemd_unit(Path("/tmp/settings.toml"))

        self.assertIn("Description=TransClip dictation service", unit)
        self.assertIn("-m transclip.cli --settings /tmp/settings.toml serve", unit)
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
                    "transclip.daemon_lifecycle.install_shortcut",
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
            patch("transclip.daemon.paste_capability", return_value=capability),
            patch("transclip.daemon.clipboard_capability", return_value=clipboard_capability),
        ):
            status = collect_status(Settings(port=0), runtime=FakeRuntime(system="Other"))

        self.assertFalse(status["paste"]["ok"])
        self.assertIn("wtype unusable", status["paste"]["detail"])


if __name__ == "__main__":
    unittest.main()
