import unittest
from unittest.mock import patch

import numpy as np
from granite_speach.audio import recording_debug, sounddevice_summary
from granite_speach.settings import Settings


class FakeRecorder:
    samples = np.array([[0], [1000], [-1000]], dtype=np.int16)

    def __init__(self, settings):
        self.settings = settings
        self.started = False

    def start(self):
        self.started = True

    def stop_samples(self):
        return type(self).samples


class AudioDebugTests(unittest.TestCase):
    def test_recording_debug_reports_audio_metrics(self):
        with patch("granite_speach.audio.time.sleep"):
            result = recording_debug(Settings(sample_rate=3), recorder_cls=FakeRecorder)

        self.assertEqual(result["sample_rate"], 3)
        self.assertEqual(result["channel_count"], 1)
        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["duration"], 1.0)
        self.assertEqual(result["peak_amplitude"], 1000.0)
        self.assertFalse(result["silent"])

    def test_recording_debug_reports_silence(self):
        FakeRecorder.samples = np.zeros((4, 1), dtype=np.int16)
        try:
            with patch("granite_speach.audio.time.sleep"):
                result = recording_debug(Settings(sample_rate=4), recorder_cls=FakeRecorder)
        finally:
            FakeRecorder.samples = np.array([[0], [1000], [-1000]], dtype=np.int16)

        self.assertTrue(result["silent"])
        self.assertEqual(result["rms_amplitude"], 0.0)

    def test_sounddevice_summary_handles_missing_dependency(self):
        with patch.dict("sys.modules", {"sounddevice": None}):
            self.assertEqual(sounddevice_summary(), "sounddevice unavailable")


if __name__ == "__main__":
    unittest.main()
