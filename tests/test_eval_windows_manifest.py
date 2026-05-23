import json
import unittest
from pathlib import Path


class EvalWindowsManifestTests(unittest.TestCase):
    def test_windows_manifest_has_relaxed_thresholds_and_granite_candidates(self):
        path = Path(__file__).resolve().parents[1] / "eval" / "windows" / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["platform"], "windows-x86_64")
        self.assertGreaterEqual(payload["thresholds"]["release_to_ready_p95_ms"], 6000)
        backends = {candidate["asr_backend"] for candidate in payload["candidates"]}
        self.assertEqual(backends, {"granite"})
        self.assertNotIn("granite_nar", backends)
        self.assertGreaterEqual(len(payload["cases"]), 1)


if __name__ == "__main__":
    unittest.main()
