import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from granite_speach import daemon
from granite_speach.daemon import (
    SERVICE_NAME,
    append_toggle_log,
    build_systemd_unit,
    install_linux_daemon,
    last_toggle_log_event,
    toggle_log_path,
)


class DaemonTests(unittest.TestCase):
    def test_systemd_unit_uses_current_python_and_cli_serve(self):
        unit = build_systemd_unit(Path("/tmp/settings.toml"))

        self.assertIn("Description=Granite Speach dictation service", unit)
        self.assertIn("-m granite_speach.cli --settings /tmp/settings.toml serve", unit)
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
                "command": "/bin/sh -lc granite-speach",
            },
        )()

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch("granite_speach.daemon_lifecycle.Path.home", return_value=home),
                patch(
                    "granite_speach.daemon_lifecycle.install_gnome_shortcut",
                    return_value=shortcut,
                ) as install_shortcut,
            ):
                results = install_linux_daemon(runner=runner)

            unit_path = home / ".config" / "systemd" / "user" / SERVICE_NAME
            self.assertTrue(unit_path.exists())
            self.assertIn("granite_speach.cli serve", unit_path.read_text(encoding="utf-8"))
            self.assertIn(["systemctl", "--user", "daemon-reload"], calls)
            self.assertIn(["systemctl", "--user", "enable", "--now", SERVICE_NAME], calls)
            install_shortcut.assert_called_once()
            self.assertTrue(all(result.ok for result in results))

    def test_toggle_log_is_jsonl_and_last_event_is_decoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toggle-record.log"
            append_toggle_log({"action": "started"}, path)
            append_toggle_log({"action": "stopped", "text": "hello"}, path)

            self.assertEqual(last_toggle_log_event(path), {"action": "stopped", "text": "hello"})

    def test_toggle_log_path_is_under_cache_dir_on_linux(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("granite_speach.daemon_lifecycle.platform.system", return_value="Linux"),
            patch("granite_speach.daemon_lifecycle.Path.home", return_value=Path(tmp)),
        ):
            self.assertEqual(
                toggle_log_path(),
                Path(tmp) / ".cache" / "granite-speach" / "toggle-record.log",
            )

    def test_paste_status_reports_probe_failure(self):
        capability = type(
            "Capability",
            (),
            {"ok": False, "backend": None, "detail": "wtype unusable: unsupported"},
        )()
        with patch("granite_speach.daemon.paste_capability", return_value=capability):
            status = daemon._paste_status()

        self.assertFalse(status["ok"])
        self.assertIn("wtype unusable", status["detail"])


if __name__ == "__main__":
    unittest.main()
