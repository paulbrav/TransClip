import io
import unittest
from urllib.error import URLError
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from granite_speach.cli import main


class FakeClient:
    result = {"status": "recording", "action": "started"}

    def __init__(self, settings):
        self.settings = settings

    def record_toggle(self):
        return type(self).result


class CliTests(unittest.TestCase):
    def test_toggle_record_starts_without_paste(self):
        FakeClient.result = {"status": "recording", "action": "started"}
        stdout = io.StringIO()
        with (
            patch("granite_speach.cli.InferenceClient", FakeClient),
            patch("granite_speach.cli.paste_transcript") as paste,
            redirect_stdout(stdout),
        ):
            code = main(["toggle-record", "--paste"])

        self.assertEqual(code, 0)
        self.assertIn('"action": "started"', stdout.getvalue())
        paste.assert_not_called()

    def test_toggle_record_stops_and_pastes_transcript(self):
        FakeClient.result = {"status": "ready", "action": "stopped", "text": "hello"}
        paste_result = type(
            "PasteResult",
            (),
            {
                "pasted": True,
                "restored": True,
                "transcript_left_on_clipboard": False,
                "error_detail": None,
            },
        )()
        stdout = io.StringIO()
        with (
            patch("granite_speach.cli.InferenceClient", FakeClient),
            patch("granite_speach.cli.paste_transcript", return_value=paste_result) as paste,
            redirect_stdout(stdout),
        ):
            code = main(["toggle-record", "--paste"])

        self.assertEqual(code, 0)
        paste.assert_called_once()
        self.assertEqual(paste.call_args.args[0], "hello")
        self.assertIn('"paste":', stdout.getvalue())
        self.assertIn('"pasted": true', stdout.getvalue())

    def test_toggle_record_reports_unavailable_service(self):
        class DownClient(FakeClient):
            def record_toggle(self):
                raise URLError("connection refused")

        stderr = io.StringIO()
        with (
            patch("granite_speach.cli.InferenceClient", DownClient),
            patch("granite_speach.cli.notify") as notify,
            redirect_stderr(stderr),
        ):
            code = main(["toggle-record", "--paste"])

        self.assertEqual(code, 1)
        self.assertIn("Granite service is not running", stderr.getvalue())
        notify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
