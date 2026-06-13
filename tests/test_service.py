import base64
import json
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from transclip.cleanup import FaithfulRuleCleanupBackend
from transclip.history import read_history
from transclip.service import InferenceEngine
from transclip.settings import Settings

from tests.service_helpers import FakeASR, FakeRecorder, FakeTextBackend, http_json, serve_test_engine, stop_server


class FakeWaveformASR(FakeASR):
    def __init__(self):
        super().__init__()
        self.waveform_lengths = []
        self.waveform_sample_rates = []

    def transcribe_waveform(self, waveform, sample_rate=16000):
        self.waveform_lengths.append(len(waveform))
        self.waveform_sample_rates.append(sample_rate)
        return self.transcribe(Path("waveform.wav"), keywords=[])


class FakeMlxASR(FakeASR):
    name = "mlx-audio"

    def __init__(self):
        super().__init__()
        self.durations = []
        self.non_silent = []

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None):
        with wave.open(str(wav_path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            self.durations.append(wav.getnframes() / wav.getframerate())
            self.non_silent.append(any(frames))
        return super().transcribe(wav_path, keywords=keywords)


class FakeStopEvent:
    def __init__(self):
        self.wait_calls = 0
        self._set = False

    def wait(self, timeout):
        del timeout
        self.wait_calls += 1
        return self._set

    def is_set(self):
        return self._set

    def set(self):
        self._set = True


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self._history_tmp = tempfile.TemporaryDirectory()
        self._history_patch = patch(
            "transclip.history.history_path",
            return_value=Path(self._history_tmp.name) / "history.jsonl",
        )
        self._history_patch.start()

    def tearDown(self):
        self._history_patch.stop()
        self._history_tmp.cleanup()

    def test_engine_health_and_transcribe(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            engine = InferenceEngine(
                Settings(),
                asr_backend=FakeASR(),
                cleanup_backend=FaithfulRuleCleanupBackend(),
            )

            health = engine.health()
            result = engine.transcribe(wav)
            keyword_result = engine.transcribe(wav, keywords=["PyTorch", "ROCm"])

            self.assertEqual(health["status"], "ready")
            self.assertEqual(result["raw_asr"], "hello from ROCm")
            self.assertEqual(result["text"], "Hello from ROCm.")
            self.assertEqual(keyword_result["text"], "Hello from ROCm.")
            self.assertEqual(engine.asr_backend.keywords, ["PyTorch", "ROCm"])
            self.assertIn("end_to_end", result["timings_ms"])

    def test_engine_warms_asr_only_when_requested(self):
        lazy_asr = FakeASR()
        InferenceEngine(
            Settings(),
            asr_backend=lazy_asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        self.assertEqual(lazy_asr.calls, [])

        warm_asr = FakeASR()
        InferenceEngine(
            Settings(sample_rate=16000),
            asr_backend=warm_asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
            warm_asr=True,
        )

        self.assertEqual(len(warm_asr.calls), 1)
        self.assertEqual(warm_asr.keywords, [])

    def test_engine_warm_asr_uses_non_silent_audio(self):
        class InspectingASR(FakeASR):
            def transcribe(self, wav_path: Path, keywords: list[str] | None = None):
                with wave.open(str(wav_path), "rb") as wav:
                    self.sample_rate = wav.getframerate()
                    self.frames = wav.readframes(wav.getnframes())
                return super().transcribe(wav_path, keywords=keywords)

        asr = InspectingASR()
        InferenceEngine(
            Settings(sample_rate=8000),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
            warm_asr=True,
        )

        self.assertEqual(asr.sample_rate, 8000)
        self.assertTrue(any(asr.frames))

    def test_engine_warms_mlx_audio_buckets_from_1_to_12_seconds(self):
        asr = FakeMlxASR()
        InferenceEngine(
            Settings(sample_rate=16000),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
            warm_asr=True,
        )

        self.assertEqual(asr.durations, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        self.assertEqual(asr.keywords, [])
        self.assertTrue(all(asr.non_silent))

    def test_engine_rewarms_speech_after_startup_mlx_audio_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            speech_wav = Path(tmp) / "speech-warmup.wav"
            _write_test_wav(speech_wav, seconds=1.0, sample_rate=16000)
            asr = FakeMlxASR()

            with patch(
                "transclip.service.engine._mlx_speech_warmup_wav",
                return_value=speech_wav,
            ) as speech_warmup:
                InferenceEngine(
                    Settings(
                        sample_rate=16000,
                        asr_backend="mlx_audio_whisper",
                    ),
                    asr_backend=asr,
                    cleanup_backend=FaithfulRuleCleanupBackend(),
                    warm_asr=True,
                )

        speech_warmup.assert_called_once()
        self.assertEqual(asr.calls[-1], speech_wav)
        self.assertEqual(asr.durations, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0])

    def test_engine_warms_remaining_bucket_shapes(self):
        asr = FakeWaveformASR()
        engine = InferenceEngine(
            Settings(sample_rate=16000, warm_bucket_shapes_s=8),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )

        engine.warm_bucket_shapes(threading.Event())

        self.assertEqual(asr.waveform_lengths, [64_000, 96_000, 128_000])
        self.assertEqual(asr.waveform_sample_rates, [16_000, 16_000, 16_000])

    def test_engine_skips_background_mlx_audio_buckets_after_startup(self):
        asr = FakeMlxASR()
        engine = InferenceEngine(
            Settings(sample_rate=16000),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )

        engine.warm_bucket_shapes(threading.Event())

        self.assertEqual(asr.durations, [])

    def test_background_mlx_warmup_does_not_wait_while_dictation_is_busy(self):
        asr = FakeMlxASR()
        engine = InferenceEngine(
            Settings(sample_rate=16000),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        statuses = ["transcribing", "ready"]
        engine.dictation_session.status = lambda: statuses.pop(0) if statuses else "ready"
        stop_event = FakeStopEvent()

        engine.warm_bucket_shapes(stop_event)

        self.assertEqual(stop_event.wait_calls, 0)
        self.assertEqual(asr.durations, [])

    def test_background_mlx_warmup_exits_when_stopped(self):
        asr = FakeMlxASR()
        engine = InferenceEngine(
            Settings(sample_rate=16000),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        stop_event = threading.Event()
        stop_event.set()

        engine.warm_bucket_shapes(stop_event)

        self.assertEqual(asr.durations, [])

    def test_bucket_warmup_waits_while_recording(self):
        asr = FakeWaveformASR()
        engine = InferenceEngine(
            Settings(sample_rate=16000, warm_bucket_shapes_s=4),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        statuses = ["recording", "ready"]
        engine.dictation_session.status = lambda: statuses.pop(0) if statuses else "ready"
        stop_event = FakeStopEvent()

        engine.warm_bucket_shapes(stop_event)

        self.assertEqual(stop_event.wait_calls, 1)
        self.assertEqual(asr.waveform_lengths, [64_000])

    def test_bucket_warmup_exits_when_stopped(self):
        asr = FakeWaveformASR()
        engine = InferenceEngine(
            Settings(sample_rate=16000, warm_bucket_shapes_s=4),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        stop_event = threading.Event()
        stop_event.set()

        engine.warm_bucket_shapes(stop_event)

        self.assertEqual(asr.waveform_lengths, [])

    def test_bucket_warmup_skips_file_only_asr_backend(self):
        asr = FakeASR()
        engine = InferenceEngine(
            Settings(sample_rate=16000, warm_bucket_shapes_s=4),
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )

        engine.warm_bucket_shapes(threading.Event())

        self.assertEqual(asr.calls, [])

    def test_cleanup_text_uses_model_cleanup_when_always_on(self):
        text_backend = FakeTextBackend(["Model cleaned via /cleanup"])
        engine = InferenceEngine(
            Settings(voice_model_cleanup_always_on=True),
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
            text_backend=text_backend,
        )

        result = engine.cleanup_text("hello ,world")

        self.assertEqual(result["text"], "Model cleaned via /cleanup")
        self.assertEqual(result["backend"], "fake-text:fake-model")

    def test_health_reports_dictation_cleanup_mode(self):
        rule_engine = InferenceEngine(
            Settings(),
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        model_engine = InferenceEngine(
            Settings(voice_model_cleanup_always_on=True),
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
            text_backend=FakeTextBackend(["unused"]),
        )

        self.assertEqual(rule_engine.health()["dictation_cleanup"], "rule")
        self.assertEqual(rule_engine.health()["cleanup_backend"], "rule-based")
        self.assertEqual(model_engine.health()["dictation_cleanup"], "model")
        self.assertEqual(model_engine.health()["cleanup_backend"], "fake-text:fake-model")

    def test_health_and_transcribe_agree_on_cleanup_backend(self):
        text_backend = FakeTextBackend(["Model cleaned"])
        engine = InferenceEngine(
            Settings(voice_model_cleanup_always_on=True),
            asr_backend=FakeASR("hello ,world"),
            cleanup_backend=FaithfulRuleCleanupBackend(),
            text_backend=text_backend,
        )
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            transcribed = engine.transcribe(wav)

        self.assertEqual(
            engine.health()["cleanup_backend"],
            transcribed["cleanup_backend"],
        )

    def test_http_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            settings = Settings(host="127.0.0.1", port=0)
            engine = InferenceEngine(
                settings,
                asr_backend=FakeASR(),
                cleanup_backend=FaithfulRuleCleanupBackend(),
            )
            server, thread, host, port = serve_test_engine(settings, engine)
            base_url = f"http://{host}:{port}"
            try:
                health = http_json("GET", f"{base_url}/health")
                cleaned = http_json("POST", f"{base_url}/cleanup", {"text": "hello ,world"})
                transcribed = http_json("POST", f"{base_url}/transcribe", {"audio_path": str(wav)})
                transcribed_clean = http_json(
                    "POST",
                    f"{base_url}/cleanup/transcribe",
                    {"audio_base64": base64.b64encode(wav.read_bytes()).decode("ascii")},
                )
                missing = http_json("POST", f"{base_url}/transcribe", {})

                self.assertEqual(health["status"], "ready")
                self.assertEqual(cleaned["text"], "Hello, world.")
                self.assertEqual(transcribed["text"], "Hello from ROCm.")
                self.assertEqual(transcribed_clean["text"], "Hello from ROCm.")
                self.assertIn("Request must include audio_path", missing["error"])
                events = read_history(path=Path(self._history_tmp.name) / "history.jsonl")
                self.assertEqual(events[0]["source"], "/cleanup/transcribe")
                self.assertEqual(events[1]["source"], "/transcribe")
            finally:
                stop_server(server, thread)

    def test_http_base64_transcription_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            asr = FakeASR()
            engine = InferenceEngine(
                Settings(host="127.0.0.1", port=0),
                asr_backend=asr,
                cleanup_backend=FaithfulRuleCleanupBackend(),
            )
            server, thread, host, port = serve_test_engine(engine.settings, engine)
            try:
                response = http_json(
                    "POST",
                    f"http://{host}:{port}/cleanup/transcribe",
                    {"audio_base64": base64.b64encode(wav.read_bytes()).decode("ascii")},
                )
            finally:
                stop_server(server, thread)

            self.assertEqual(response["text"], "Hello from ROCm.")
            self.assertFalse(asr.wav_path.exists())

    def test_history_write_failure_does_not_fail_transcription(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            engine = InferenceEngine(
                Settings(),
                asr_backend=FakeASR(),
                cleanup_backend=FaithfulRuleCleanupBackend(),
            )
            with patch("transclip.service.engine.append_transcript_history", side_effect=OSError("history full")):
                result = engine.transcribe(wav, record_history=True)

            self.assertEqual(result["text"], "Hello from ROCm.")
            self.assertIn("history full", result["history_error"])

    def test_http_record_start_and_stop_transcribes_service_audio(self):
        settings = Settings(host="127.0.0.1", port=0)
        asr = FakeASR()
        engine = InferenceEngine(
            settings,
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        server, thread, host, port = serve_test_engine(settings, engine)
        base_url = f"http://{host}:{port}"
        try:
            with patch("transclip.service.engine.AudioRecorder", FakeRecorder):
                started = http_json("POST", f"{base_url}/record/start", {})
                health = http_json("GET", f"{base_url}/health")
                stopped = http_json("POST", f"{base_url}/record/stop", {"cleanup": True})

            self.assertEqual(started["status"], "recording")
            self.assertEqual(health["status"], "recording")
            self.assertEqual(stopped["text"], "Hello from ROCm.")
            self.assertIn("duration_ms", stopped)
            self.assertTrue(asr.wav_path.name.endswith(".wav"))
        finally:
            stop_server(server, thread)

    def test_http_health_reports_transcribing_after_recording_stops(self):
        transcribe_entered = threading.Event()
        transcribe_release = threading.Event()

        class BlockingASR(FakeASR):
            def transcribe(self, wav_path: Path, keywords: list[str] | None = None):
                transcribe_entered.set()
                transcribe_release.wait(timeout=2)
                return super().transcribe(wav_path, keywords=keywords)

        settings = Settings(host="127.0.0.1", port=0)
        asr = BlockingASR()
        engine = InferenceEngine(
            settings,
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        server, thread, host, port = serve_test_engine(settings, engine)
        base_url = f"http://{host}:{port}"
        try:
            with patch("transclip.service.engine.AudioRecorder", FakeRecorder):
                http_json("POST", f"{base_url}/record/start", {})
                stopped = {}
                stop_thread = threading.Thread(
                    target=lambda: stopped.update(
                        http_json("POST", f"{base_url}/record/stop", {"cleanup": True})
                    )
                )
                stop_thread.start()
                self.assertTrue(transcribe_entered.wait(timeout=2))

                health = http_json("GET", f"{base_url}/health")
                self.assertEqual(health["status"], "transcribing")

                transcribe_release.set()
                stop_thread.join(timeout=2)

            self.assertEqual(stopped["text"], "Hello from ROCm.")
            self.assertEqual(http_json("GET", f"{base_url}/health")["status"], "ready")
        finally:
            transcribe_release.set()
            stop_server(server, thread)

    def test_http_record_stop_can_discard_short_recording(self):
        settings = Settings(host="127.0.0.1", port=0)
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        server, thread, host, port = serve_test_engine(settings, engine)
        base_url = f"http://{host}:{port}"
        try:
            with patch("transclip.service.engine.AudioRecorder", FakeRecorder):
                http_json("POST", f"{base_url}/record/start", {})
                stopped = http_json("POST", f"{base_url}/record/stop", {"discard": True})

            self.assertEqual(stopped["status"], "ready")
            self.assertTrue(stopped["discarded"])
        finally:
            stop_server(server, thread)

    def test_http_record_toggle_starts_and_stops(self):
        settings = Settings(
            host="127.0.0.1",
            port=0,
            min_recording_ms=0,
            toggle_cooldown_ms=0,
        )
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        server, thread, host, port = serve_test_engine(settings, engine)
        base_url = f"http://{host}:{port}"
        try:
            with patch("transclip.service.engine.AudioRecorder", FakeRecorder):
                started = http_json("POST", f"{base_url}/record/toggle", {})
                stopped = http_json("POST", f"{base_url}/record/toggle", {"cleanup": True})

            self.assertEqual(started["status"], "recording")
            self.assertEqual(started["action"], "started")
            self.assertEqual(stopped["status"], "ready")
            self.assertEqual(stopped["action"], "stopped")
            self.assertEqual(stopped["text"], "Hello from ROCm.")
            self.assertIn("duration_ms", stopped)
        finally:
            stop_server(server, thread)

    def test_http_record_toggle_discards_under_minimum_duration(self):
        settings = Settings(
            host="127.0.0.1",
            port=0,
            min_recording_ms=10_000,
            toggle_cooldown_ms=0,
        )
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        server, thread, host, port = serve_test_engine(settings, engine)
        base_url = f"http://{host}:{port}"
        try:
            with patch("transclip.service.engine.AudioRecorder", FakeRecorder):
                http_json("POST", f"{base_url}/record/toggle", {})
                stopped = http_json("POST", f"{base_url}/record/toggle", {})

            self.assertEqual(stopped["status"], "ready")
            self.assertEqual(stopped["action"], "discarded")
            self.assertTrue(stopped["discarded"])
            self.assertIn("duration_ms", stopped)
        finally:
            stop_server(server, thread)

    def test_debug_capture_writes_error_logs_for_http_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                host="127.0.0.1",
                port=0,
                debug_capture=True,
                debug_capture_dir=str(Path(tmp) / "captures"),
            )
            engine = InferenceEngine(
                settings,
                asr_backend=FakeASR(),
                cleanup_backend=FaithfulRuleCleanupBackend(),
            )
            server, thread, host, port = serve_test_engine(settings, engine)
            try:
                response = http_json("POST", f"http://{host}:{port}/transcribe", {})
                capture_dir = Path(response["debug_capture_dir"])

                self.assertIn("Request must include audio_path", response["error"])
                self.assertTrue((capture_dir / "error.log").exists())
                self.assertTrue((capture_dir / "error.json").exists())
                self.assertIn("http_request", (capture_dir / "error.log").read_text())
            finally:
                stop_server(server, thread)

    def test_record_partial_returns_empty_when_not_recording(self):
        settings = Settings(host="127.0.0.1", port=0)
        server, thread, host, port = serve_test_engine(settings)
        try:
            payload = http_json("GET", f"http://{host}:{port}/record/partial")
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["partial_text"], "")
        finally:
            stop_server(server, thread)

    def test_record_partial_returns_text_while_recording(self):
        from transclip.service.streaming import StreamingDictationAdapter

        from tests.service_helpers import FakeStreamingASR, FakeStreamingSessionFactory

        streaming = FakeStreamingASR()
        factory = FakeStreamingSessionFactory(streaming)
        settings = Settings(host="127.0.0.1", port=0)
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        streaming_adapter = StreamingDictationAdapter(
            settings,
            factory,
            engine.process_asr_result,
        )
        engine._streaming = streaming_adapter
        engine.dictation_session = engine.dictation_session.__class__(
            settings,
            transcribe=engine._transcribe_for_session,
            streaming=streaming_adapter,
        )

        class FakeChunkedRecorder(FakeRecorder):
            def __init__(self, recorder_settings, *, on_chunk=None, **kwargs):
                super().__init__(recorder_settings)

            def stop_capture(self):
                self.discarded = True

        server, thread, host, port = serve_test_engine(settings, engine)
        try:
            with patch("transclip.service.streaming.ChunkedAudioRecorder", FakeChunkedRecorder):
                engine.start_recording()
            streaming.feed(b"\x00" * 200)
            payload = http_json("GET", f"http://{host}:{port}/record/partial")
            self.assertEqual(payload["status"], "recording")
            self.assertEqual(payload["partial_text"], "word")
        finally:
            stop_server(server, thread)

    def test_record_toggle_cooldown_ignores_immediate_second_toggle(self):
        settings = Settings(
            host="127.0.0.1",
            port=0,
            min_recording_ms=0,
            toggle_cooldown_ms=500,
        )
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        with patch("transclip.service.engine.AudioRecorder", FakeRecorder):
            started = engine.toggle_recording()
            ignored = engine.toggle_recording()

        self.assertEqual(started["action"], "started")
        self.assertEqual(ignored["status"], "recording")
        self.assertEqual(ignored["action"], "ignored")
        self.assertEqual(ignored["reason"], "toggle_cooldown")
        self.assertEqual(ignored["cooldown_ms"], 500)

    def test_normal_dictation_remains_rule_cleanup_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            text_backend = FakeTextBackend(["model cleanup"])
            engine = InferenceEngine(
                Settings(voice_model_cleanup_always_on=False),
                asr_backend=FakeASR("hello ,world"),
                cleanup_backend=FaithfulRuleCleanupBackend(),
                text_backend=text_backend,
            )

            result = engine.transcribe(wav)

            self.assertEqual(result["text"], "Hello, world.")
            self.assertEqual(text_backend.messages, [])
            self.assertEqual(result["voice_mode"], "dictation")

    def test_normal_dictation_does_not_call_text_model(self):
        class ExplodingTextBackend:
            name = "exploding-text"
            model_name = "should-not-load"

            def generate(self, messages, *, max_new_tokens):
                raise AssertionError("normal dictation should not call the text model")

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            engine = InferenceEngine(
                Settings(voice_model_cleanup_always_on=False),
                asr_backend=FakeASR("hello ,world"),
                cleanup_backend=FaithfulRuleCleanupBackend(),
                text_backend=ExplodingTextBackend(),
            )

            result = engine.transcribe(wav)

            self.assertEqual(result["text"], "Hello, world.")
            self.assertEqual(result["voice_mode"], "dictation")
            self.assertIsNone(result["shell"])

    def test_always_on_model_cleanup_applies_to_normal_dictation(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            text_backend = FakeTextBackend(["Model cleaned normal dictation"])
            engine = InferenceEngine(
                Settings(voice_model_cleanup_always_on=True),
                asr_backend=FakeASR("hello ,world"),
                cleanup_backend=FaithfulRuleCleanupBackend(),
                text_backend=text_backend,
            )

            result = engine.transcribe(wav)

            self.assertEqual(result["text"], "Model cleaned normal dictation")
            self.assertEqual(result["cleanup_backend"], "fake-text:fake-model")

    def test_shell_trigger_returns_diagnostic_when_text_model_fails(self):
        class FailingTextBackend:
            name = "failing-text"
            model_name = "missing-model"

            def generate(self, messages, *, max_new_tokens):
                raise RuntimeError("text model unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            engine = InferenceEngine(
                Settings(shellcheck_enabled=False),
                asr_backend=FakeASR("shell command list files"),
                cleanup_backend=FaithfulRuleCleanupBackend(),
                text_backend=FailingTextBackend(),
            )

            result = engine.transcribe(wav)

            self.assertEqual(result["voice_mode"], "shell")
            self.assertIs(result["submit"], False)
            self.assertFalse(result["shell"]["valid"])
            self.assertEqual(result["shell"]["command"], "")
            self.assertIn("model generation failed", result["shell"]["diagnostics"][0])
            self.assertTrue(result["text"].startswith("# TransClip could not produce valid Bash"))

    def test_debug_capture_writes_voice_and_shell_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            engine = InferenceEngine(
                Settings(
                    shellcheck_enabled=False,
                    debug_capture=True,
                    debug_capture_dir=str(Path(tmp) / "captures"),
                ),
                asr_backend=FakeASR("shell command list files"),
                cleanup_backend=FaithfulRuleCleanupBackend(),
                text_backend=FakeTextBackend(['{"command": "ls -la"}']),
            )

            result = engine.transcribe(wav)

            metadata_path = Path(result["debug_capture_dir"]) / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["voice_mode"], "shell")
            self.assertEqual(metadata["voice_trigger"], "shell command")
            self.assertEqual(metadata["shell"]["command"], "ls -la")

def _write_test_wav(path: Path, *, seconds: float, sample_rate: int) -> None:
    frame_count = int(seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x01\x00" * frame_count)


if __name__ == "__main__":
    unittest.main()
