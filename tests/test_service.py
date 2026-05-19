import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from granite_speach.cleanup import FaithfulRuleCleanupBackend
from granite_speach.history import read_history
from granite_speach.service import InferenceEngine
from granite_speach.settings import Settings
from tests.service_helpers import FakeASR, FakeRecorder, http_json, serve_test_engine, stop_server


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self._history_tmp = tempfile.TemporaryDirectory()
        self._history_patch = patch(
            "granite_speach.history.history_path",
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

    def test_http_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            settings = Settings(host="127.0.0.1", port=0, cleanup_runtime="test_rule")
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
                Settings(host="127.0.0.1", port=0, cleanup_runtime="test_rule"),
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
                Settings(cleanup_runtime="test_rule"),
                asr_backend=FakeASR(),
                cleanup_backend=FaithfulRuleCleanupBackend(),
            )
            with patch("granite_speach.service.append_transcript_history", side_effect=OSError("history full")):
                result = engine.transcribe(wav, record_history=True)

            self.assertEqual(result["text"], "Hello from ROCm.")
            self.assertIn("history full", result["history_error"])

    def test_http_record_start_and_stop_transcribes_service_audio(self):
        settings = Settings(host="127.0.0.1", port=0, cleanup_runtime="test_rule")
        asr = FakeASR()
        engine = InferenceEngine(
            settings,
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        server, thread, host, port = serve_test_engine(settings, engine)
        base_url = f"http://{host}:{port}"
        try:
            with patch("granite_speach.service.AudioRecorder", FakeRecorder):
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

    def test_http_record_stop_can_discard_short_recording(self):
        settings = Settings(host="127.0.0.1", port=0, cleanup_runtime="test_rule")
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        server, thread, host, port = serve_test_engine(settings, engine)
        base_url = f"http://{host}:{port}"
        try:
            with patch("granite_speach.service.AudioRecorder", FakeRecorder):
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
            cleanup_runtime="test_rule",
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
            with patch("granite_speach.service.AudioRecorder", FakeRecorder):
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
            cleanup_runtime="test_rule",
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
            with patch("granite_speach.service.AudioRecorder", FakeRecorder):
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
                cleanup_runtime="test_rule",
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

    def test_record_toggle_cooldown_ignores_immediate_second_toggle(self):
        settings = Settings(
            host="127.0.0.1",
            port=0,
            cleanup_runtime="test_rule",
            min_recording_ms=0,
            toggle_cooldown_ms=500,
        )
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        with patch("granite_speach.service.AudioRecorder", FakeRecorder):
            started = engine.toggle_recording()
            ignored = engine.toggle_recording()

        self.assertEqual(started["action"], "started")
        self.assertEqual(ignored["status"], "recording")
        self.assertEqual(ignored["action"], "ignored")
        self.assertEqual(ignored["reason"], "toggle_cooldown")
        self.assertEqual(ignored["cooldown_ms"], 500)


if __name__ == "__main__":
    unittest.main()
