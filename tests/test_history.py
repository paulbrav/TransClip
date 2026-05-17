import tempfile
import unittest
from pathlib import Path

from granite_speach.history import append_history_event, append_transcript_history, read_history
from granite_speach.settings import Settings


class HistoryTests(unittest.TestCase):
    def test_append_read_latest_limit_and_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            append_history_event({"timestamp": "1", "text": "first", "source": "/transcribe"}, path)
            path.write_text(path.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")
            append_history_event({"timestamp": "2", "text": "second", "source": "/record/toggle"}, path)

            events = read_history(limit=1, path=path)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["text"], "second")

    def test_empty_transcripts_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            append_history_event({"text": "   "}, path)

            self.assertFalse(path.exists())

    def test_append_transcript_history_includes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            append_transcript_history(
                {
                    "text": "hello",
                    "raw_asr": "hello",
                    "asr_backend": "fake",
                    "asr_model": "fake-model",
                    "cleanup_backend": "rule",
                    "cleanup_enabled": True,
                },
                Settings(),
                source="/record/stop",
                duration_ms=123.4,
                path=path,
            )

            event = read_history(path=path)[0]
            self.assertEqual(event["source"], "/record/stop")
            self.assertEqual(event["duration_ms"], 123.4)
            self.assertEqual(event["asr_backend"], "fake")


if __name__ == "__main__":
    unittest.main()
