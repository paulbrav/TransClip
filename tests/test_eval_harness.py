import json
import tempfile
import unittest
from pathlib import Path

from granite_speach.eval_harness import (
    cleanup_drift_delta,
    is_cleanup_semantic_drift,
    keyword_preservation,
    run_eval,
    word_error_rate,
)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, cleanup=None, keywords=None):
        self.calls.append({"audio_path": Path(audio_path).name, "cleanup": cleanup, "keywords": keywords})
        name = Path(audio_path).name
        if name == "warm.wav":
            return {
                "text": "Warmup should not appear.",
                "raw_asr": "warmup should not appear",
                "timings_ms": {"end_to_end": 999.0},
            }
        return {
            "text": "PyTorch on ROCm.",
            "raw_asr": "pytorch on rocm",
            "timings_ms": {"end_to_end": 250.0},
        }


class EvalHarnessTests(unittest.TestCase):
    def test_word_error_rate(self):
        self.assertEqual(word_error_rate("hello world", "hello world"), 0.0)
        self.assertEqual(word_error_rate("hello world", "hello"), 0.5)

    def test_keyword_preservation(self):
        score = keyword_preservation("PyTorch on ROCm", ["PyTorch", "ROCm", "MLX"])
        self.assertAlmostEqual(score, 2 / 3)

    def test_cleanup_drift_delta_only_counts_regression(self):
        self.assertEqual(cleanup_drift_delta(0.5, 0.25), 0.0)
        self.assertAlmostEqual(cleanup_drift_delta(0.25, 0.5), 0.25)
        self.assertIsNone(cleanup_drift_delta(None, 0.5))
        self.assertFalse(is_cleanup_semantic_drift(0.05))
        self.assertTrue(is_cleanup_semantic_drift(0.051))

    def test_eval_runs_warmup_cases_outside_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "warmup_cases": [{"audio_path": "warm.wav", "cleanup": True}],
                        "cases": [
                            {
                                "audio_path": "measured.wav",
                                "reference": "PyTorch on ROCm.",
                                "keywords": ["PyTorch", "ROCm"],
                                "cleanup": True,
                                "paste_attempted": True,
                                "paste_success": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            engine = FakeEngine()
            result = run_eval(manifest, engine)
        self.assertEqual(result["summary"]["warmup_cases"], 1)
        self.assertEqual(
            engine.calls,
            [
                {"audio_path": "warm.wav", "cleanup": True, "keywords": None},
                {"audio_path": "measured.wav", "cleanup": True, "keywords": ["PyTorch", "ROCm"]},
            ],
        )
        self.assertEqual(result["summary"]["cases"], 1)
        self.assertEqual(result["summary"]["cleanup_semantic_drift_failures"], 0)
        self.assertEqual(len(result["results"]), 1)
        case = result["results"][0]
        self.assertTrue(case["audio_path"].endswith("measured.wav"))
        self.assertEqual(case["text"], "PyTorch on ROCm.")
        self.assertEqual(case["raw_asr"], "pytorch on rocm")
        self.assertEqual(case["wer"], 0.0)
        self.assertAlmostEqual(case["raw_asr_wer"], 1 / 3)
        self.assertEqual(case["cleanup_drift_wer_delta"], 0.0)
        self.assertEqual(case["keyword_preservation"], 1.0)
        self.assertEqual(case["timings_ms"]["end_to_end"], 250.0)
        self.assertEqual(case["paste_attempted"], True)
        self.assertEqual(case["paste_success"], True)
        self.assertEqual(result["summary"]["paste_attempts"], 1)
        self.assertEqual(result["summary"]["paste_successes"], 1)
        self.assertEqual(result["summary"]["paste_failures"], 0)

if __name__ == "__main__":
    unittest.main()
