import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from granite_speach.recording_ops import toggle_recording
from granite_speach.settings import Settings


class FakeClient:
    base_url = "http://service"

    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def record_toggle(self):
        if self.error:
            raise self.error
        return dict(self.result)


class FakePasteResult:
    copied = True
    pasted = False
    restored = False
    transcript_left_on_clipboard = True
    clipboard_backend = "fake-clipboard"
    paste_backend = None
    error_detail = "fake paste failed"


class RecordingOpsTests(unittest.TestCase):
    def setUp(self):
        self._log_patch = patch("granite_speach.recording_ops.append_toggle_log")
        self._log_patch.start()

    def tearDown(self):
        self._log_patch.stop()

    def test_service_unavailable_is_renderable_error(self):
        outcome = toggle_recording(
            Settings(),
            client=FakeClient(error=URLError("refused")),
        )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.notification_message, "Granite service is not running.")

    def test_http_rejection_is_renderable_error(self):
        outcome = toggle_recording(
            Settings(),
            client=FakeClient(error=HTTPError("http://service", 500, "boom", None, None)),
        )

        self.assertFalse(outcome.ok)
        self.assertIn("HTTP 500", outcome.notification_message)

    def test_started_and_discarded_do_not_expose_latest_transcript(self):
        started = toggle_recording(Settings(), client=FakeClient({"action": "started", "status": "recording"}))
        discarded = toggle_recording(Settings(), client=FakeClient({"action": "discarded", "status": "ready"}))

        self.assertEqual(started.latest_transcript, "")
        self.assertEqual(discarded.latest_transcript, "")

    def test_stopped_paste_failure_carries_transcript_and_message(self):
        with patch("granite_speach.recording_ops.paste_transcript", return_value=FakePasteResult()):
            outcome = toggle_recording(
                Settings(),
                paste=True,
                client=FakeClient({"action": "stopped", "status": "ready", "text": "Hello."}),
            )

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.latest_transcript, "Hello.")
        self.assertIn("still on the clipboard", outcome.notification_message)
        self.assertFalse(outcome.payload["paste"]["pasted"])


if __name__ == "__main__":
    unittest.main()
