import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tauri_paste_smoke.py"
SPEC = importlib.util.spec_from_file_location("tauri_paste_smoke", SCRIPT_PATH)
tauri_paste_smoke = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(tauri_paste_smoke)


class TauriPasteSmokeTests(unittest.TestCase):
    def test_validate_smoke_payload_accepts_paste_and_restore(self):
        result = tauri_paste_smoke.validate_smoke_payload(
            {
                "ok": True,
                "userAgent": "tauri-webkit",
                "insertedText": tauri_paste_smoke.SMOKE_TEXT,
                "clipboardAfter": "previous",
                "previousClipboard": "previous",
                "errorText": "",
            }
        )

        self.assertEqual(result["status"], "pass")

    def test_validate_smoke_payload_rejects_missing_inserted_text(self):
        result = tauri_paste_smoke.validate_smoke_payload(
            {
                "ok": True,
                "insertedText": "",
                "clipboardAfter": "previous",
                "previousClipboard": "previous",
                "errorText": "",
            }
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("focused field", result["error"])

    def test_validate_smoke_payload_rejects_unrestored_clipboard(self):
        result = tauri_paste_smoke.validate_smoke_payload(
            {
                "ok": True,
                "insertedText": tauri_paste_smoke.SMOKE_TEXT,
                "clipboardAfter": tauri_paste_smoke.SMOKE_TEXT,
                "previousClipboard": "previous",
                "errorText": "",
            }
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("clipboard", result["error"])


if __name__ == "__main__":
    unittest.main()
