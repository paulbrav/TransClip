import base64
import importlib.util
from pathlib import Path
import struct
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tauri_recorder_smoke.py"
SPEC = importlib.util.spec_from_file_location("tauri_recorder_smoke", SCRIPT_PATH)
tauri_recorder_smoke = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(tauri_recorder_smoke)


def wav_bytes(sample_rate=48_000, samples=4):
    data = b"\0\0" * samples
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(data)),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16),
            b"data",
            struct.pack("<I", len(data)),
            data,
        ]
    )


class TauriRecorderSmokeTests(unittest.TestCase):
    def test_validate_smoke_payload_accepts_mono_pcm_wav(self):
        payload = {
            "ok": True,
            "userAgent": "tauri-webkit",
            "wavBase64": base64.b64encode(wav_bytes()).decode("ascii"),
        }

        result = tauri_recorder_smoke.validate_smoke_payload(payload)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["channels"], 1)
        self.assertEqual(result["sample_rate"], 48_000)
        self.assertEqual(result["bits_per_sample"], 16)

    def test_validate_smoke_payload_reports_recorder_error(self):
        result = tauri_recorder_smoke.validate_smoke_payload(
            {"ok": False, "error": "NotAllowedError: The request is not allowed"}
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("NotAllowedError", result["error"])


if __name__ == "__main__":
    unittest.main()
