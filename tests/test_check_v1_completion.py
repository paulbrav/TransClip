import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from granite_speach.doctor import Check

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_v1_completion.py"
SPEC = importlib.util.spec_from_file_location("check_v1_completion", SCRIPT_PATH)
check_v1_completion = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(check_v1_completion)


def eval_payload(cases=25):
    return {
        "summary": {
            "cases": cases,
            "mean_release_to_ready_ms": 300.0,
            "under_700ms": cases,
            "under_1500ms": cases,
            "mean_keyword_preservation": 1.0,
            "mean_wer": 0.1,
            "cleanup_semantic_drift_failures": 0,
            "paste_failures": 0,
        },
        "results": [
            {
                "audio_path": f"case_{index:02d}.wav",
                "text": "ok",
                "raw_asr": "ok",
                "reference": "ok",
                "wer": 0.1,
                "raw_asr_wer": 0.1,
                "cleanup_drift_wer_delta": 0.0,
                "cleanup_semantic_drift": False,
                "keyword_preservation": 1.0,
                "timings_ms": {"end_to_end": 300.0},
            }
            for index in range(cases)
        ],
    }


class CheckV1CompletionTests(unittest.TestCase):
    def test_blocks_without_real_eval_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic = root / "synthetic.json"
            synthetic.write_text(json.dumps(eval_payload(25)), encoding="utf-8")

            report = check_v1_completion.check_completion(
                real_results_path=root / "missing-real.json",
                synthetic_results_path=synthetic,
                doctor_checks=[Check("paste_tools", True, "ok")],
            )

            self.assertEqual(report["status"], "blocked")
            self.assertIn("real-usage eval results missing", report["blockers"][0])

    def test_blocks_on_failed_doctor_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic = root / "synthetic.json"
            real = root / "real.json"
            synthetic.write_text(json.dumps(eval_payload(25)), encoding="utf-8")
            real.write_text(json.dumps(eval_payload(20)), encoding="utf-8")

            report = check_v1_completion.check_completion(
                real_results_path=real,
                synthetic_results_path=synthetic,
                doctor_checks=[Check("paste_tools", False, "wtype unusable")],
            )

            self.assertEqual(report["status"], "blocked")
            self.assertIn("doctor paste_tools failed", report["blockers"][0])

    def test_passes_with_real_eval_and_doctor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic = root / "synthetic.json"
            real = root / "real.json"
            synthetic.write_text(json.dumps(eval_payload(25)), encoding="utf-8")
            real.write_text(json.dumps(eval_payload(20)), encoding="utf-8")

            report = check_v1_completion.check_completion(
                real_results_path=real,
                synthetic_results_path=synthetic,
                doctor_checks=[Check("paste_tools", True, "ok")],
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["blockers"], [])


if __name__ == "__main__":
    unittest.main()
