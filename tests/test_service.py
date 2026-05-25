import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.cleanup import FaithfulRuleCleanupBackend
from transclip.history import read_history
from transclip.service import InferenceEngine
from transclip.settings import Settings

from tests.service_helpers import FakeASR, FakeRecorder, FakeTextBackend, http_json, serve_test_engine, stop_server


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


if __name__ == "__main__":
    unittest.main()
