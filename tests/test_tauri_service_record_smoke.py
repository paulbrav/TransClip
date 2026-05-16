import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tauri_service_record_smoke.py"
SPEC = importlib.util.spec_from_file_location("tauri_service_record_smoke", SCRIPT_PATH)
tauri_service_record_smoke = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(tauri_service_record_smoke)

HOTKEY_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tauri_global_hotkey_smoke.py"
HOTKEY_SPEC = importlib.util.spec_from_file_location("tauri_global_hotkey_smoke", HOTKEY_SCRIPT_PATH)
tauri_global_hotkey_smoke = importlib.util.module_from_spec(HOTKEY_SPEC)
assert HOTKEY_SPEC and HOTKEY_SPEC.loader
HOTKEY_SPEC.loader.exec_module(tauri_global_hotkey_smoke)


class TauriServiceRecordSmokeTests(unittest.TestCase):
    def test_validate_smoke_payload_accepts_recording_flow(self):
        result = tauri_service_record_smoke.validate_smoke_payload(
            {
                "ok": True,
                "userAgent": "tauri-webkit",
                "statusWhileRecording": "Recording",
                "stopEnabledWhileRecording": True,
                "finalStatus": "Ready",
                "errorText": "",
            }
        )

        self.assertEqual(result["status"], "pass")

    def test_validate_smoke_payload_rejects_missing_recording_status(self):
        result = tauri_service_record_smoke.validate_smoke_payload(
            {
                "ok": True,
                "statusWhileRecording": "Ready",
                "stopEnabledWhileRecording": True,
                "finalStatus": "Ready",
                "errorText": "",
            }
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("Recording", result["error"])

    def test_validate_smoke_payload_rejects_disabled_stop_control(self):
        result = tauri_service_record_smoke.validate_smoke_payload(
            {
                "ok": True,
                "statusWhileRecording": "Recording",
                "stopEnabledWhileRecording": False,
                "finalStatus": "Ready",
                "errorText": "",
            }
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("Stop control", result["error"])

    def test_hotkey_fake_service_reset_clears_counters(self):
        tauri_global_hotkey_smoke.FakeServiceHandler.health_count = 2
        tauri_global_hotkey_smoke.FakeServiceHandler.start_count = 1
        tauri_global_hotkey_smoke.FakeServiceHandler.stop_count = 1
        tauri_global_hotkey_smoke.FakeServiceHandler.recording = True

        tauri_global_hotkey_smoke.FakeServiceHandler.reset()

        self.assertEqual(tauri_global_hotkey_smoke.FakeServiceHandler.health_count, 0)
        self.assertEqual(tauri_global_hotkey_smoke.FakeServiceHandler.start_count, 0)
        self.assertEqual(tauri_global_hotkey_smoke.FakeServiceHandler.stop_count, 0)
        self.assertFalse(tauri_global_hotkey_smoke.FakeServiceHandler.recording)


if __name__ == "__main__":
    unittest.main()
