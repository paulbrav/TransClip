import http.client
import json
import unittest

from transclip.cleanup import FaithfulRuleCleanupBackend
from transclip.service import InferenceEngine
from transclip.settings import Settings

from tests.service_helpers import FakeASR, serve_test_engine, stop_server


def _get(host: str, port: int, path: str) -> tuple[int, dict]:
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", path, headers={"content-type": "application/json"})
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        return response.status, (json.loads(raw) if raw else {})
    finally:
        conn.close()


class ServiceReadinessTest(unittest.TestCase):
    def test_readyz_reports_ready_with_200(self) -> None:
        server, thread, host, port = serve_test_engine()
        self.addCleanup(stop_server, server, thread)
        status, payload = _get(host, port, "/readyz")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ready"])
        self.assertFalse(payload["env_broken"])
        self.assertIsNone(payload["error"])

    def test_healthz_is_an_alias_for_readyz(self) -> None:
        server, thread, host, port = serve_test_engine()
        self.addCleanup(stop_server, server, thread)
        status, payload = _get(host, port, "/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ready"])

    def test_readyz_reports_503_when_ml_stack_missing(self) -> None:
        # Reproduce the original failure as a LOUD signal: warm_asr() raised
        # ModuleNotFoundError (torch pruned), so the service serves but reports
        # not-ready via 503 instead of silently 500-ing every transcription.
        settings = Settings(host="127.0.0.1", port=0)
        engine = InferenceEngine(
            settings,
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
        )
        engine.asr_ready = False
        engine.asr_env_broken = True
        engine.asr_last_error = "ModuleNotFoundError: No module named 'torch'"

        server, thread, host, port = serve_test_engine(settings, engine)
        self.addCleanup(stop_server, server, thread)
        status, payload = _get(host, port, "/readyz")
        self.assertEqual(status, 503)
        self.assertFalse(payload["ready"])
        self.assertTrue(payload["env_broken"])
        self.assertIn("torch", payload["error"])
