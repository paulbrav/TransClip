import unittest
from dataclasses import dataclass
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from transclip.recording_ops import toggle_recording
from transclip.settings import Settings


class FakeClient:
    base_url = "http://service"

    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def record_toggle(self):
        if self.error:
            raise self.error
        return dict(self.result)


@dataclass
class FakePasteResult:
    copied: bool = True
    pasted: bool = False
    injected: bool = False
    restored: bool = False
    transcript_left_on_clipboard: bool = True
    clipboard_backend: str = "fake-clipboard"
    paste_backend: str | None = None
    error_detail: str = "fake paste failed"
    paste_shortcut: str = "Ctrl+Shift+V"
    delivery: str = "inject"
    focused_app_kind: str | None = None


class RecordingOpsTests(unittest.TestCase):
    def setUp(self):
        self._log_patch = patch("transclip.recording_ops.append_toggle_log")
        self._log_patch.start()

    def tearDown(self):
        self._log_patch.stop()

    def test_service_unavailable_is_renderable_error(self):
        outcome = toggle_recording(
            Settings(),
            client=FakeClient(error=URLError("refused")),
        )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.notification_message, "TransClip service is not running.")

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
        with patch("transclip.recording_ops.paste_transcript", return_value=FakePasteResult()):
            outcome = toggle_recording(
                Settings(),
                paste=True,
                client=FakeClient({"action": "stopped", "status": "ready", "text": "Hello."}),
            )

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.latest_transcript, "Hello.")
        self.assertIn("still on the clipboard", outcome.notification_message)
        self.assertFalse(outcome.payload["paste"]["pasted"])

    def test_stopped_paste_logs_delivery_metadata(self):
        paste_result = FakePasteResult(
            pasted=True,
            injected=True,
            paste_shortcut="Ctrl+Shift+V",
            delivery="inject",
            focused_app_kind="terminal",
            error_detail="",
        )
        with (
            patch("transclip.recording_ops.paste_transcript", return_value=paste_result),
            patch("transclip.recording_ops._append_toggle_log") as append_log,
        ):
            toggle_recording(
                Settings(),
                paste=True,
                client=FakeClient({"action": "stopped", "status": "ready", "text": "Hello."}),
            )

        logged = append_log.call_args[0][0]
        self.assertEqual(logged["paste"]["paste_shortcut"], "Ctrl+Shift+V")
        self.assertEqual(logged["paste"]["delivery"], "inject")
        self.assertEqual(logged["paste"]["focused_app_kind"], "terminal")

    def test_clipboard_only_paste_sets_notice_message(self):
        paste_result = FakePasteResult(
            pasted=True,
            injected=False,
            delivery="clipboard_only",
            paste_backend=None,
            error_detail="",
        )
        with patch("transclip.recording_ops.paste_transcript", return_value=paste_result):
            outcome = toggle_recording(
                Settings(text_delivery_mode="clipboard_only"),
                paste=True,
                client=FakeClient({"action": "stopped", "status": "ready", "text": "Hello."}),
            )

        self.assertEqual(outcome.paste_notice_message, "Transcript copied to clipboard. Paste manually.")
        self.assertEqual(outcome.notification_message, outcome.paste_notice_message)


if __name__ == "__main__":
    unittest.main()
