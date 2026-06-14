import http.client
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.cleanup import FaithfulRuleCleanupBackend
from transclip.service import InferenceEngine
from transclip.settings import Settings

from tests.service_helpers import serve_test_engine, stop_server


def _get(host: str, port: int, path: str) -> tuple[int, dict]:
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", path, headers={"content-type": "application/json"})
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        return response.status, (json.loads(raw) if raw else {})
    finally:
        conn.close()


class _RaisingASR:
    """ASR backend whose warmup transcribe raises, to drive the readiness classifier."""

    name = "raising-asr"
    model = "raising-model"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def transcribe(self, wav_path, keywords=None):
        raise self._exc


def _warmup_engine(exc: Exception, *, stack_importable: bool) -> InferenceEngine:
    settings = Settings(host="127.0.0.1", port=0)
    # Patch the import probe: .venv-dev has no torch, so without this every case
    # would classify as env_broken. Patching lets us test both branches.
    with patch("transclip.service.engine._ml_stack_importable", return_value=stack_importable):
        return InferenceEngine(
            settings,
            asr_backend=_RaisingASR(exc),
            cleanup_backend=FaithfulRuleCleanupBackend(),
            warm_asr=True,
        )


class ServiceReadinessRouteTest(unittest.TestCase):
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

    def test_readyz_returns_503_end_to_end_when_warmup_failed(self) -> None:
        # Reproduce the original outage as a LOUD signal: warmup failed with the
        # ML stack missing, so the service serves but /readyz reports 503 instead
        # of silently 500-ing every transcription.
        engine = _warmup_engine(ModuleNotFoundError("No module named 'torch'"), stack_importable=False)
        server, thread, host, port = serve_test_engine(engine.settings, engine)
        self.addCleanup(stop_server, server, thread)
        status, payload = _get(host, port, "/readyz")
        self.assertEqual(status, 503)
        self.assertFalse(payload["ready"])
        self.assertTrue(payload["env_broken"])
        self.assertIn("torch", payload["error"])


class AsrReadinessClassificationTest(unittest.TestCase):
    def test_module_not_found_is_env_broken(self) -> None:
        engine = _warmup_engine(ModuleNotFoundError("No module named 'torch'"), stack_importable=False)
        self.assertFalse(engine.asr_ready)
        self.assertTrue(engine.asr_env_broken)
        self.assertIn("torch", engine.asr_last_error)

    def test_runtime_error_with_torch_missing_is_env_broken(self) -> None:
        # AR backend + asr_device=cuda: resolve_torch_device raises a bare
        # RuntimeError (no ImportError in the chain) even though torch is gone.
        # Must still be classified env-broken, via the import probe -- this is the
        # regression guard for the misclassification the review caught.
        engine = _warmup_engine(
            RuntimeError("CUDA/ROCm was requested, but torch cannot execute a GPU tensor operation"),
            stack_importable=False,
        )
        self.assertFalse(engine.asr_ready)
        self.assertTrue(engine.asr_env_broken)

    def test_failure_with_stack_present_is_recoverable(self) -> None:
        # Stack imports fine but warmup failed (e.g. weights not downloaded):
        # recoverable, not env-broken.
        engine = _warmup_engine(RuntimeError("model weights not found"), stack_importable=True)
        self.assertFalse(engine.asr_ready)
        self.assertFalse(engine.asr_env_broken)

    def test_wire_error_redacts_home_path(self) -> None:
        home = str(Path.home())
        engine = _warmup_engine(OSError(f"cannot read {home}/.cache/huggingface/x"), stack_importable=True)
        # Full error retained internally for the journal log...
        self.assertIn(home, engine.asr_last_error)
        # ...but the /readyz wire payload redacts the home path/username.
        wire_error = str(engine.asr_readiness()["error"])
        self.assertNotIn(home, wire_error)
        self.assertIn("~/.cache/huggingface", wire_error)
