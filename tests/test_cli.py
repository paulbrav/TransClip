import io
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests.platform_helpers import linux_runtime
from tests.service_helpers import FakeRecorder, serve_test_engine, stop_server
from transclip.cli import main
from transclip.settings import Settings, write_settings


class InMemoryClipboard:
    backend_name = "fake-clipboard"
    value = ""

    def read(self) -> str:
        return type(self).value

    def write(self, text: str) -> None:
        type(self).value = text


class FakePasteInjector:
    pasted = True

    def __init__(self):
        self.backend_name = None

    def paste(self) -> bool:
        if type(self).pasted:
            self.backend_name = "fake-paste"
            return True
        return False

    def error_detail(self) -> str:
        return "fake paste failed"


class CliTests(unittest.TestCase):
    def test_toggle_record_starts_without_paste(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, thread, host, port = serve_test_engine(transcript="hello")
            settings_path = write_test_settings(root, host, port)
            try:
                stdout = io.StringIO()
                with (
                    patch("transclip.service.AudioRecorder", FakeRecorder),
                    patch("transclip.paste.SystemClipboard", InMemoryClipboard),
                    patch("transclip.paste.SystemPasteInjector", FakePasteInjector),
                    patch("transclip.cli_commands.notify"),
                    patch("transclip.daemon_lifecycle.Path.home", return_value=root),
                    redirect_stdout(stdout),
                ):
                    code = main(["--settings", str(settings_path), "toggle-record", "--paste"])
            finally:
                stop_server(server, thread)

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["action"], "started")
            self.assertEqual(payload["status"], "recording")
            self.assertNotIn("paste", payload)
            log_event = read_last_toggle_log(root)
            self.assertEqual(log_event["action"], "started")

    def test_toggle_record_stops_and_pastes_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, thread, host, port = serve_test_engine(transcript="hello")
            settings_path = write_test_settings(root, host, port)
            try:
                InMemoryClipboard.value = "prior"
                with (
                    patch("transclip.service.AudioRecorder", FakeRecorder),
                    patch("transclip.paste.SystemClipboard", InMemoryClipboard),
                    patch("transclip.paste.SystemPasteInjector", FakePasteInjector),
                    patch("transclip.cli_commands.notify"),
                    patch("transclip.daemon_lifecycle.Path.home", return_value=root),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(main(["--settings", str(settings_path), "toggle-record"]), 0)

                stdout = io.StringIO()
                with (
                    patch("transclip.service.AudioRecorder", FakeRecorder),
                    patch("transclip.paste.SystemClipboard", InMemoryClipboard),
                    patch("transclip.paste.SystemPasteInjector", FakePasteInjector),
                    patch("transclip.cli_commands.notify"),
                    patch("transclip.daemon_lifecycle.Path.home", return_value=root),
                    redirect_stdout(stdout),
                ):
                    code = main(["--settings", str(settings_path), "toggle-record", "--paste"])
            finally:
                stop_server(server, thread)

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["action"], "stopped")
            self.assertEqual(payload["text"], "Hello.")
            self.assertTrue(payload["paste"]["pasted"])
            self.assertEqual(payload["paste"]["clipboard_backend"], "fake-clipboard")
            self.assertEqual(payload["paste"]["paste_backend"], "fake-paste")
            self.assertEqual(InMemoryClipboard.value, "Hello.")
            log_event = read_last_toggle_log(root)
            self.assertEqual(log_event["action"], "stopped")
            self.assertEqual(log_event["text"], "Hello.")

    def test_toggle_record_reports_unavailable_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_path = write_test_settings(root, "127.0.0.1", unused_local_port())
            stderr = io.StringIO()
            with (
                patch("transclip.cli_commands.notify"),
                patch("transclip.daemon_lifecycle.Path.home", return_value=root),
                redirect_stderr(stderr),
            ):
                code = main(["--settings", str(settings_path), "toggle-record", "--paste"])

            self.assertEqual(code, 1)
            self.assertIn("TransClip service is not running", stderr.getvalue())
            log_event = read_last_toggle_log(root)
            self.assertEqual(log_event["action"], "error")
            self.assertEqual(log_event["error"], "TransClip service is not running.")

    def test_toggle_record_log_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".cache").write_text("not a directory", encoding="utf-8")
            server, thread, host, port = serve_test_engine(transcript="hello")
            settings_path = write_test_settings(root, host, port)
            try:
                stdout = io.StringIO()
                with (
                    patch("transclip.service.AudioRecorder", FakeRecorder),
                    patch("transclip.daemon_lifecycle.Path.home", return_value=root),
                    redirect_stdout(stdout),
                ):
                    code = main(["--settings", str(settings_path), "toggle-record"])
            finally:
                stop_server(server, thread)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["action"], "started")
        self.assertIn("log_error", payload)
        self.assertIn(".cache", payload["log_error"])

    def test_history_json_and_copy(self):
        stdout = io.StringIO()
        events = [{"text": "latest", "timestamp": "now", "source": "/transcribe"}]
        with (
            patch("transclip.cli_commands.read_history", return_value=events),
            redirect_stdout(stdout),
        ):
            code = main(["history", "--json"])

        self.assertEqual(code, 0)
        self.assertIn('"text": "latest"', stdout.getvalue())

        class Clipboard:
            text = ""

            def write(self, text):
                type(self).text = text

        stdout = io.StringIO()
        with (
            patch("transclip.cli_commands.read_history", return_value=events),
            patch("transclip.cli_commands.SystemClipboard", Clipboard),
            redirect_stdout(stdout),
        ):
            code = main(["history", "--copy", "1"])

        self.assertEqual(code, 0)
        self.assertEqual(Clipboard.text, "latest")

    def test_config_get_set_uses_settings_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--settings", str(path), "config", "set", "toggle_cooldown_ms", "250"])
            self.assertEqual(code, 0)
            self.assertIn("toggle_cooldown_ms", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--settings", str(path), "config", "get", "toggle_cooldown_ms"])
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue().strip(), "250")

    def test_models_list_uses_local_catalog(self):
        stdout = io.StringIO()
        with (
            patch("transclip.runtime_profile.get_runtime", return_value=linux_runtime()),
            redirect_stdout(stdout),
        ):
            code = main(["models", "list"])

        self.assertEqual(code, 0)
        self.assertIn("ibm-granite/granite-speech-4.1-2b-nar", stdout.getvalue())

    def test_install_gnome_shortcut_uses_configured_hotkey(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.toml"
            write_settings(Settings(hotkey_linux="<Control><Alt>space"), settings_path)
            stdout = io.StringIO()
            shortcut = type(
                "Shortcut",
                (),
                {
                    "name": "TransClip Toggle",
                    "path": "/shortcut/",
                    "binding": "<Control><Alt>space",
                    "command": "/bin/sh -lc transclip",
                },
            )()
            with (
                patch("transclip.cli_commands.install_shortcut", return_value=shortcut) as install,
                redirect_stdout(stdout),
            ):
                code = main(["--settings", str(settings_path), "install-gnome-shortcut"])

        self.assertEqual(code, 0)
        self.assertEqual(install.call_args.kwargs["binding"], "<Control><Alt>space")
        self.assertIn("Binding: <Control><Alt>space", stdout.getvalue())

    def test_tray_command_runs_python_tray(self):
        with (
            patch("transclip.tray.get_runtime", return_value=linux_runtime()),
            patch("transclip.tray.run_python_tray", return_value=7),
        ):
            code = main(["tray"])

        self.assertEqual(code, 7)


def write_test_settings(root: Path, host: str, port: int, **overrides) -> Path:
    settings = Settings(
        host=host,
        port=port,
        cleanup_runtime="test_rule",
        min_recording_ms=0,
        toggle_cooldown_ms=0,
        clipboard_restore_delay_ms=0,
        **overrides,
    )
    path = root / "settings.toml"
    write_settings(settings, path)
    return path


def read_last_toggle_log(root: Path) -> dict:
    log_path = root / ".cache" / "transclip" / "toggle-record.log"
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


def unused_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
