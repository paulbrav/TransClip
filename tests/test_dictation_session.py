import unittest
from pathlib import Path

from granite_speach.dictation_session import DictationSession
from granite_speach.settings import Settings
from tests.service_helpers import FakeRecorder


class StepClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


class DictationSessionTests(unittest.TestCase):
    def test_toggle_discards_recording_under_minimum_duration(self):
        session = DictationSession(
            Settings(min_recording_ms=500, toggle_cooldown_ms=0),
            transcribe=lambda _wav, _cleanup, _source: {"text": "should not run"},
            recorder_factory=FakeRecorder,
            clock=StepClock([1.0, 1.0, 1.1, 1.1, 1.1]),
        )

        started = session.toggle_recording()
        stopped = session.toggle_recording()

        self.assertEqual(started["action"], "started")
        self.assertEqual(stopped["action"], "discarded")
        self.assertTrue(stopped["discarded"])
        self.assertEqual(stopped["duration_ms"], 100.0)

    def test_stop_calls_transcriber_with_recorded_wav_and_source(self):
        calls = []

        def transcribe(wav_path: Path, cleanup, source):
            calls.append((wav_path, cleanup, source, wav_path.exists()))
            return {"text": "Hello.", "status": "ready"}

        session = DictationSession(
            Settings(min_recording_ms=0, toggle_cooldown_ms=0),
            transcribe=transcribe,
            recorder_factory=FakeRecorder,
            clock=StepClock([2.0, 2.25]),
        )

        session.start_recording()
        result = session.stop_recording(cleanup=True, source="/record/stop")

        self.assertEqual(result["text"], "Hello.")
        self.assertEqual(result["duration_ms"], 250.0)
        self.assertEqual(calls[0][1:], (True, "/record/stop", True))


if __name__ == "__main__":
    unittest.main()
