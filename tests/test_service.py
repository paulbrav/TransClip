from pathlib import Path
import base64
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib import request

from granite_speach.asr import TranscriptionResult
from granite_speach.cleanup import FaithfulRuleCleanupBackend
from granite_speach.service import InferenceEngine, create_server
from granite_speach.settings import Settings


class FakeASR:
    name = "fake"
    model = "fake-model"

    def transcribe(self, wav_path: Path, keywords: list[str]) -> TranscriptionResult:
        self.wav_path = wav_path
        self.keywords = keywords
        return TranscriptionResult("hello from ROCm", {"asr": 1.0}, self.name, self.model)


class FakeRecorder:
    def __init__(self, settings):
        self.settings = settings
        self.started = False

    def start(self):
        self.started = True

    def stop_to_wav(self, output_path: Path):
        output_path.write_bytes(b"not really wav")
        return output_path


class ServiceTests(unittest.TestCase):
    def test_engine_health_and_transcribe(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            engine = InferenceEngine(
                Settings(),
                asr_backend=FakeASR(),
                cleanup_backend=FaithfulRuleCleanupBackend(),
                keywords=["ROCm"],
            )

            health = engine.health()
            result = engine.transcribe(wav)

            self.assertEqual(health["status"], "ready")
            self.assertEqual(result["raw_asr"], "hello from ROCm")
            self.assertEqual(result["text"], "Hello from ROCm.")
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
                keywords=["ROCm"],
            )
            server = create_server(settings, engine)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
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
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_engine_loads_keywords_from_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            keyword_file = Path(tmp) / "keywords.txt"
            keyword_file.write_text("Tauri\nROCm\n", encoding="utf-8")
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            asr = FakeASR()
            engine = InferenceEngine(
                Settings(),
                asr_backend=asr,
                cleanup_backend=FaithfulRuleCleanupBackend(),
                keyword_path=keyword_file,
            )

            engine.transcribe(wav, cleanup=False)

            self.assertEqual(asr.keywords, ["Tauri", "ROCm"])

    def test_transcribe_can_override_keywords_per_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            wav.write_bytes(b"not really wav")
            asr = FakeASR()
            engine = InferenceEngine(
                Settings(),
                asr_backend=asr,
                cleanup_backend=FaithfulRuleCleanupBackend(),
                keywords=["ROCm"],
            )

            engine.transcribe(wav, cleanup=False, keywords=["Tauri"])

            self.assertEqual(asr.keywords, ["Tauri"])

    def test_http_record_start_and_stop_transcribes_service_audio(self):
        settings = Settings(host="127.0.0.1", port=0, cleanup_runtime="test_rule")
        asr = FakeASR()
        engine = InferenceEngine(
            settings,
            asr_backend=asr,
            cleanup_backend=FaithfulRuleCleanupBackend(),
            keywords=["ROCm"],
        )
        server = create_server(settings, engine)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
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
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_record_stop_can_discard_short_recording(self):
        settings = Settings(host="127.0.0.1", port=0, cleanup_runtime="test_rule")
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
            keywords=["ROCm"],
        )
        server = create_server(settings, engine)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        base_url = f"http://{host}:{port}"
        try:
            with patch("granite_speach.service.AudioRecorder", FakeRecorder):
                http_json("POST", f"{base_url}/record/start", {})
                stopped = http_json("POST", f"{base_url}/record/stop", {"discard": True})

            self.assertEqual(stopped["status"], "ready")
            self.assertTrue(stopped["discarded"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

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
                keywords=["ROCm"],
            )
            server = create_server(settings, engine)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            try:
                response = http_json("POST", f"http://{host}:{port}/transcribe", {})
                capture_dir = Path(response["debug_capture_dir"])

                self.assertIn("Request must include audio_path", response["error"])
                self.assertTrue((capture_dir / "error.log").exists())
                self.assertTrue((capture_dir / "error.json").exists())
                self.assertIn("http_request", (capture_dir / "error.log").read_text())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


def http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method=method,
    )
    try:
        with request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        if hasattr(exc, "read"):
            return json.loads(exc.read().decode("utf-8"))
        raise


if __name__ == "__main__":
    unittest.main()
