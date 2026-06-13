import unittest
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from transclip.asr import TranscriptionResult
from transclip.asr_streaming import PartialTranscript
from transclip.service import DictationSession
from transclip.service.streaming import StreamingDictationAdapter
from transclip.settings import Settings

from tests.service_helpers import FakeRecorder, FakeStreamingASR, FakeStreamingSessionFactory


class StepClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


class SequencedStreamingFactory:
    def __init__(self, sessions):
        self.sessions = list(sessions)

    def __call__(self):
        return self.sessions.pop(0)


class BlockingStreamingASR:
    name = "blocking-streaming"
    model = "blocking-streaming-model"

    def __init__(self, final_text):
        self.final_text = final_text
        self.chunks = []
        self.closed = False

    @property
    def partial_text(self):
        return PartialTranscript("")

    def feed(self, pcm16_mono):
        self.chunks.append(pcm16_mono)

    def finish(self):
        return TranscriptionResult(self.final_text, {}, self.name, self.model)

    def close(self):
        self.closed = True


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

    def test_toggle_discards_recording_over_maximum_duration_without_transcribing(self):
        transcribe_calls = []

        session = DictationSession(
            Settings(max_recording_ms=1_000, min_recording_ms=0, toggle_cooldown_ms=0),
            transcribe=lambda _wav, _cleanup, _source: transcribe_calls.append(_wav),
            recorder_factory=FakeRecorder,
            clock=StepClock([1.0, 1.0, 2.25, 2.25, 2.25]),
        )

        session.toggle_recording()
        stopped = session.toggle_recording()

        self.assertEqual(stopped["action"], "discarded")
        self.assertTrue(stopped["discarded"])
        self.assertEqual(stopped["reason"], "max_recording_duration")
        self.assertEqual(stopped["max_recording_ms"], 1_000)
        self.assertEqual(transcribe_calls, [])

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

    def test_status_moves_from_stopping_to_transcribing_while_stop_finishes(self):
        stop_entered = Event()
        stop_release = Event()
        transcribe_entered = Event()
        transcribe_release = Event()

        class BlockingRecorder(FakeRecorder):
            def stop_to_wav(self, output_path: Path):
                stop_entered.set()
                stop_release.wait(timeout=2)
                return super().stop_to_wav(output_path)

        def transcribe(wav_path: Path, cleanup, source):
            del wav_path, cleanup, source
            transcribe_entered.set()
            transcribe_release.wait(timeout=2)
            return {"text": "Hello.", "status": "ready"}

        session = DictationSession(
            Settings(min_recording_ms=0, toggle_cooldown_ms=0),
            transcribe=transcribe,
            recorder_factory=BlockingRecorder,
            clock=lambda: 1.0,
        )
        session.start_recording()

        stopped = {}
        stop_thread = Thread(target=lambda: stopped.update(session.stop_recording()))
        stop_thread.start()

        self.assertTrue(stop_entered.wait(timeout=2))
        self.assertEqual(session.status(), "stopping")

        stop_release.set()
        self.assertTrue(transcribe_entered.wait(timeout=2))
        self.assertEqual(session.status(), "transcribing")

        transcribe_release.set()
        stop_thread.join(timeout=2)

        self.assertEqual(stopped["text"], "Hello.")
        self.assertEqual(session.status(), "ready")

    def test_streaming_finish_uses_adapter_instead_of_wav_transcriber(self):
        streaming = FakeStreamingASR(final_text="streamed final")
        factory = FakeStreamingSessionFactory(streaming)
        settings = Settings(min_recording_ms=0, toggle_cooldown_ms=0)
        transcribe_calls = []

        def process_asr_result(asr_result, *, cleanup, source, **kwargs):
            return {
                "text": asr_result.text,
                "raw_asr": asr_result.text,
                "status": "ready",
                "cleanup": {},
                "cleanup_enabled": True,
                "timings_ms": asr_result.timings_ms,
            }

        adapter = StreamingDictationAdapter(settings, factory, process_asr_result)

        class FakeChunkedRecorder(FakeRecorder):
            def __init__(self, recorder_settings, *, on_chunk=None, **kwargs):
                super().__init__(recorder_settings)

            def stop_capture(self):
                self.discarded = True

        session = DictationSession(
            settings,
            transcribe=lambda *_args: transcribe_calls.append(True),
            streaming=adapter,
            clock=StepClock([1.0, 1.2]),
        )
        with patch("transclip.service.streaming.ChunkedAudioRecorder", FakeChunkedRecorder):
            session.start_recording()
        session.partial_text()
        streaming.feed(b"\x00" * 100)
        result = session.stop_recording()
        self.assertEqual(result["text"], "streamed final")
        self.assertEqual(transcribe_calls, [])

    def test_streaming_stop_finish_does_not_use_new_recording_session(self):
        first_stop_entered = Event()
        first_stop_release = Event()
        first_streaming = BlockingStreamingASR("first transcript")
        second_streaming = BlockingStreamingASR("second transcript")
        factory = SequencedStreamingFactory([first_streaming, second_streaming])
        settings = Settings(min_recording_ms=0, toggle_cooldown_ms=0)

        def process_asr_result(asr_result, *, cleanup, source, **kwargs):
            return {
                "text": asr_result.text,
                "raw_asr": asr_result.text,
                "status": "ready",
                "cleanup": {},
                "cleanup_enabled": True,
                "timings_ms": asr_result.timings_ms,
            }

        adapter = StreamingDictationAdapter(settings, factory, process_asr_result)

        class FakeChunkedRecorder(FakeRecorder):
            created = 0

            def __init__(self, recorder_settings, *, on_chunk=None, **kwargs):
                super().__init__(recorder_settings)
                self.on_chunk = on_chunk
                type(self).created += 1
                self.index = type(self).created

            def stop_capture(self):
                if self.index == 1:
                    first_stop_entered.set()
                    first_stop_release.wait(timeout=2)
                self.on_chunk(f"chunk-{self.index}".encode())
                super().stop_capture()

        session = DictationSession(
            settings,
            transcribe=lambda *_args: {"text": "batch should not run"},
            streaming=adapter,
            clock=lambda: 1.0,
        )

        with patch("transclip.service.streaming.ChunkedAudioRecorder", FakeChunkedRecorder):
            session.start_recording()
            stopped = {}

            def stop_first():
                stopped.update(session.stop_recording())

            stop_thread = Thread(target=stop_first)
            stop_thread.start()
            self.assertTrue(first_stop_entered.wait(timeout=2))

            session.start_recording()
            first_stop_release.set()
            stop_thread.join(timeout=2)
            second_result = session.stop_recording()

        self.assertEqual(stopped["text"], "first transcript")
        self.assertEqual(second_result["text"], "second transcript")
        self.assertEqual(first_streaming.chunks, [b"chunk-1"])
        self.assertEqual(second_streaming.chunks, [b"chunk-2"])
        self.assertFalse(first_streaming.closed)

    def test_streaming_stop_failure_closes_detached_session(self):
        streaming = BlockingStreamingASR("unused")
        settings = Settings(min_recording_ms=0, toggle_cooldown_ms=0)
        adapter = StreamingDictationAdapter(
            settings,
            SequencedStreamingFactory([streaming]),
            lambda asr_result, **_kwargs: {"text": asr_result.text, "status": "ready"},
        )

        class FailingStopRecorder(FakeRecorder):
            def __init__(self, recorder_settings, *, on_chunk=None, **kwargs):
                super().__init__(recorder_settings)

            def stop_capture(self):
                raise RuntimeError("audio stop failed")

        session = DictationSession(
            settings,
            transcribe=lambda *_args: {"text": "batch should not run"},
            streaming=adapter,
            clock=StepClock([1.0, 1.1]),
        )

        with patch("transclip.service.streaming.ChunkedAudioRecorder", FailingStopRecorder):
            session.start_recording()
            with self.assertRaisesRegex(RuntimeError, "audio stop failed"):
                session.stop_recording()

        self.assertTrue(streaming.closed)

    def test_streaming_discard_failure_still_closes_session(self):
        streaming = BlockingStreamingASR("unused")
        settings = Settings(min_recording_ms=0, toggle_cooldown_ms=0)
        adapter = StreamingDictationAdapter(
            settings,
            SequencedStreamingFactory([streaming]),
            lambda asr_result, **_kwargs: {"text": asr_result.text, "status": "ready"},
        )

        class FailingDiscardRecorder(FakeRecorder):
            def __init__(self, recorder_settings, *, on_chunk=None, **kwargs):
                super().__init__(recorder_settings)

            def discard(self):
                raise RuntimeError("audio discard failed")

        session = DictationSession(
            settings,
            transcribe=lambda *_args: {"text": "batch should not run"},
            streaming=adapter,
            clock=StepClock([1.0, 1.1]),
        )

        with patch("transclip.service.streaming.ChunkedAudioRecorder", FailingDiscardRecorder):
            session.start_recording()
            with self.assertRaisesRegex(RuntimeError, "audio discard failed"):
                session.stop_recording(discard=True)

        self.assertTrue(streaming.closed)


if __name__ == "__main__":
    unittest.main()
