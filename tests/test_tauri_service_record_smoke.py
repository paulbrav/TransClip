import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


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

    def test_hotkey_smoke_converts_configured_shortcut_for_ydotool(self):
        self.assertEqual(
            tauri_global_hotkey_smoke.to_ydotool_hotkey("Control+Option+Space"),
            "ctrl+alt+space",
        )

    def test_hotkey_smoke_selects_evdev_backend_on_wayland(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            self.assertEqual(tauri_global_hotkey_smoke.selected_hotkey_backend(), "linux-evdev")

    def test_hotkey_smoke_selects_tauri_backend_on_x11(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=True):
            self.assertEqual(
                tauri_global_hotkey_smoke.selected_hotkey_backend(),
                "tauri-global-shortcut",
            )


if __name__ == "__main__":
    unittest.main()
