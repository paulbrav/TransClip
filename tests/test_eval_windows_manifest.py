import json
import unittest
from pathlib import Path

from transclip.eval_harness import EvalGatePolicy


def synthetic_payload(*, cases: int, latency: float) -> dict:
    return {
        "summary": {
            "cases": cases,
            "mean_release_to_ready_ms": latency,
            "under_700ms": cases if latency < 700 else 0,
            "under_1500ms": cases if latency < 1500 else 0,
            "mean_keyword_preservation": 1.0,
            "mean_wer": 0.05,
            "cleanup_semantic_drift_failures": 0,
            "paste_failures": 0,
        },
        "results": [
            {
                "audio_path": f"case_{index:02d}.wav",
                "text": "ok",
                "raw_asr": "ok",
                "reference": "ok",
                "wer": 0.05,
                "raw_asr_wer": 0.05,
                "cleanup_drift_wer_delta": 0.0,
                "cleanup_semantic_drift": False,
                "keyword_preservation": 1.0,
                "paste_attempted": None,
                "paste_success": None,
                "timings_ms": {"end_to_end": latency},
            }
            for index in range(cases)
        ],
    }


class EvalWindowsManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = Path(__file__).resolve().parents[1] / "eval" / "windows" / "manifest.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_windows_manifest_has_relaxed_thresholds_and_granite_candidates(self):
        self.assertEqual(self.manifest["platform"], "windows-x86_64")
        self.assertGreaterEqual(self.manifest["thresholds"]["release_to_ready_p95_ms"], 6000)
        backends = {candidate["asr_backend"] for candidate in self.manifest["candidates"]}
        self.assertEqual(backends, {"granite"})
        self.assertNotIn("granite_nar", backends)
        self.assertGreaterEqual(len(self.manifest["cases"]), 1)

    def test_windows_policy_loads_manifest_thresholds(self):
        policy = EvalGatePolicy.from_manifest(self.manifest_path)
        thresholds = self.manifest["thresholds"]
        case_count = len(self.manifest["cases"])

        self.assertEqual(policy.min_cases, case_count)
        self.assertEqual(policy.max_cases, case_count)
        self.assertEqual(policy.max_latency_ms, thresholds["release_to_ready_p95_ms"])
        self.assertEqual(policy.max_mean_latency_ms, thresholds["release_to_ready_p95_ms"])
        self.assertEqual(policy.max_mean_wer, thresholds["wer_max"])
        self.assertEqual(policy.min_keyword_preservation, thresholds["keyword_preservation_min"])
        self.assertEqual(policy.min_under_700_ratio, 0.0)

    def test_windows_policy_passes_fast_synthetic_payload(self):
        policy = EvalGatePolicy.from_manifest(self.manifest_path)
        report = policy.check_results(
            synthetic_payload(cases=len(self.manifest["cases"]), latency=300.0),
        )
        self.assertEqual(report["status"], "pass")

    def test_windows_policy_rejects_slow_synthetic_payload(self):
        policy = EvalGatePolicy.from_manifest(self.manifest_path)
        with self.assertRaisesRegex(ValueError, "worst latency"):
            policy.check_results(
                synthetic_payload(cases=len(self.manifest["cases"]), latency=7000.0),
            )


if __name__ == "__main__":
    unittest.main()
