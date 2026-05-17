import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch
from urllib.error import URLError

from granite_speach.cli import main


class FakeClient:
    result: ClassVar[dict[str, str]] = {"status": "recording", "action": "started"}

    def __init__(self, settings):
        self.settings = settings
        self.base_url = f"http://{settings.host}:{settings.port}"

    def record_toggle(self):
        return type(self).result


class CliTests(unittest.TestCase):
    def test_toggle_record_starts_without_paste(self):
        FakeClient.result = {"status": "recording", "action": "started"}
        stdout = io.StringIO()
        with (
            patch("granite_speach.recording_ops.InferenceClient", FakeClient),
            patch("granite_speach.recording_ops.paste_transcript") as paste,
            patch("granite_speach.recording_ops.append_toggle_log") as append_log,
            redirect_stdout(stdout),
        ):
            code = main(["toggle-record", "--paste"])

        self.assertEqual(code, 0)
        self.assertIn('"action": "started"', stdout.getvalue())
        paste.assert_not_called()
        append_log.assert_called_once()

    def test_toggle_record_stops_and_pastes_transcript(self):
        FakeClient.result = {"status": "ready", "action": "stopped", "text": "hello"}
        paste_result = type(
            "PasteResult",
            (),
            {
                "pasted": True,
                "restored": True,
                "transcript_left_on_clipboard": False,
                "copied": True,
                "clipboard_backend": "fake-clipboard",
                "paste_backend": "fake-paste",
                "error_detail": None,
            },
        )()
        stdout = io.StringIO()
        with (
            patch("granite_speach.recording_ops.InferenceClient", FakeClient),
            patch("granite_speach.recording_ops.paste_transcript", return_value=paste_result) as paste,
            patch("granite_speach.recording_ops.append_toggle_log") as append_log,
            redirect_stdout(stdout),
        ):
            code = main(["toggle-record", "--paste"])

        self.assertEqual(code, 0)
        paste.assert_called_once()
        self.assertEqual(paste.call_args.args[0], "hello")
        self.assertIn('"paste":', stdout.getvalue())
        self.assertIn('"pasted": true', stdout.getvalue())
        append_log.assert_called_once()

    def test_toggle_record_reports_unavailable_service(self):
        class DownClient(FakeClient):
            def record_toggle(self):
                raise URLError("connection refused")

        stderr = io.StringIO()
        with (
            patch("granite_speach.recording_ops.InferenceClient", DownClient),
            patch("granite_speach.cli_commands.notify") as notify,
            patch("granite_speach.recording_ops.append_toggle_log") as append_log,
            redirect_stderr(stderr),
        ):
            code = main(["toggle-record", "--paste"])

        self.assertEqual(code, 1)
        self.assertIn("Granite service is not running", stderr.getvalue())
        notify.assert_called_once()
        append_log.assert_called_once()

    def test_toggle_record_log_failure_is_nonfatal(self):
        FakeClient.result = {"status": "recording", "action": "started"}
        stdout = io.StringIO()
        with (
            patch("granite_speach.recording_ops.InferenceClient", FakeClient),
            patch("granite_speach.recording_ops.append_toggle_log", side_effect=OSError("disk full")),
            redirect_stdout(stdout),
        ):
            code = main(["toggle-record"])

        self.assertEqual(code, 0)
        self.assertIn('"action": "started"', stdout.getvalue())
        self.assertIn('"log_error": "disk full"', stdout.getvalue())

    def test_history_json_and_copy(self):
        stdout = io.StringIO()
        events = [{"text": "latest", "timestamp": "now", "source": "/transcribe"}]
        with (
            patch("granite_speach.cli_commands.read_history", return_value=events),
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
            patch("granite_speach.cli_commands.read_history", return_value=events),
            patch("granite_speach.cli_commands.SystemClipboard", Clipboard),
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
        with redirect_stdout(stdout):
            code = main(["models", "list"])

        self.assertEqual(code, 0)
        self.assertIn("ibm-granite/granite-speech-4.1-2b-nar", stdout.getvalue())

    def test_tray_command_runs_python_tray(self):
        with patch("granite_speach.tray.run_python_tray", return_value=0) as run_tray:
            code = main(["tray"])

        self.assertEqual(code, 0)
        run_tray.assert_called_once()


if __name__ == "__main__":
    unittest.main()
