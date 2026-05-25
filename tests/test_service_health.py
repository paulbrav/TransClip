import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.desktop.tray.gtk import _reexec_with_system_python
from transclip.service import build_health_status, settings_health_payload
from transclip.service.client_health import (
    fetch_service_health_result,
    service_health_check_detail,
    service_health_is_ready,
)
from transclip.service.types import ServiceHealthResponse
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime, normalize_path_text, patch_linux_gpu_runtime


class ServiceHealthTests(unittest.TestCase):
    def test_service_health_is_ready_accepts_ready_and_recording(self):
        self.assertTrue(service_health_is_ready({"status": "ready"}))
        self.assertTrue(service_health_is_ready({"status": "recording"}))
        self.assertFalse(service_health_is_ready({"status": "starting"}))
        self.assertFalse(service_health_is_ready(None))

    def test_fetch_service_health_result_returns_error_on_failure(self):
        with patch(
            "transclip.service.client.InferenceClient.health",
            side_effect=OSError("connection refused"),
        ):
            health, error = fetch_service_health_result(Settings(port=0))

        self.assertIsNone(health)
        self.assertIn("connection refused", error or "")

    def test_service_health_check_detail_includes_backend_fields(self):
        health: ServiceHealthResponse = {
            "status": "ready",
            "asr_backend": "granite",
            "cleanup_backend": "rule",
        }
        detail = service_health_check_detail(health)

        self.assertIn("status=ready", detail)
        self.assertIn("asr=granite", detail)

    def test_settings_health_payload_tracks_settings_fields(self):
        with patch_linux_gpu_runtime():
            runtime = FakeRuntime(system="Linux", home=Path("/home/test"))
            settings = Settings(voice_model_cleanup_always_on=True)

        payload = settings_health_payload(settings, runtime)

        self.assertTrue(payload["voice_model_cleanup_always_on"])
        self.assertIn("hotkey", payload)
        self.assertIn("paste_shortcut", payload)

    def test_build_health_status_returns_flat_dict(self):
        with patch_linux_gpu_runtime():
            runtime = FakeRuntime(system="Linux", home=Path("/home/test"))

        payload = build_health_status(
            status="ready",
            settings=Settings(),
            asr_backend_name="granite",
            asr_model="ibm-granite/granite-speech-4.1-2b",
            cleanup_backend="rule-based",
            dictation_cleanup="rule",
            runtime=runtime,
        )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["asr_backend"], "granite")
        self.assertIn("cleanup_enabled", payload)


class GtkTrayReexecTests(unittest.TestCase):
    def test_reexec_uses_repo_root_for_pythonpath(self):
        captured: dict[str, str] = {}

        def fake_call(command, *, cwd, env):
            captured["pythonpath"] = env["PYTHONPATH"]
            captured["cwd"] = cwd
            return 0

        with (
            patch("transclip.daemon.common.repo_root", return_value=Path("/repo/root")),
            patch("transclip.desktop.tray.gtk.sys.executable", "/venv/bin/python"),
            patch("transclip.desktop.tray.gtk.subprocess.call", side_effect=fake_call),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "resolve", lambda self: self),
        ):
            result = _reexec_with_system_python(None)

        self.assertEqual(result, 0)
        self.assertTrue(normalize_path_text(captured["pythonpath"]).startswith("/repo/root"))
        self.assertEqual(normalize_path_text(captured["cwd"]), "/repo/root")


if __name__ == "__main__":
    unittest.main()
