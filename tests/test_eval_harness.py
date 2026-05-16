import json
from pathlib import Path
import tempfile
import unittest

from granite_speach.eval_harness import (
    cleanup_drift_delta,
    is_cleanup_semantic_drift,
    keyword_preservation,
    run_keyword_ablation,
    run_eval,
    word_error_rate,
)


class FakeEngine:
    keywords = ["ROCm"]

    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, cleanup=None, keywords=None):
        self.calls.append((Path(audio_path).name, cleanup, keywords))
        return {
            "text": "PyTorch on ROCm.",
            "raw_asr": "pytorch on rocm",
            "timings_ms": {"end_to_end": 250.0},
        }


class KeywordAblationEngine:
    keywords = ["PyTorch", "ROCm"]

    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, cleanup=None, keywords=None):
        self.calls.append((Path(audio_path).name, cleanup, keywords))
        if keywords:
            text = "PyTorch on ROCm."
        else:
            text = "Pie torch on rock em."
        return {
            "text": text,
            "raw_asr": text,
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
        self.assertEqual(
            engine.calls,
            [
                ("warm.wav", True, None),
                ("measured.wav", True, ["PyTorch", "ROCm"]),
            ],
        )
        self.assertEqual(result["summary"]["warmup_cases"], 1)
        self.assertEqual(result["summary"]["cases"], 1)
        self.assertEqual(result["summary"]["cleanup_semantic_drift_failures"], 0)
        self.assertIn("raw_asr_wer", result["results"][0])
        self.assertIn("cleanup_drift_wer_delta", result["results"][0])
        self.assertEqual(result["results"][0]["paste_attempted"], True)
        self.assertEqual(result["results"][0]["paste_success"], True)
        self.assertEqual(result["summary"]["paste_attempts"], 1)
        self.assertEqual(result["summary"]["paste_successes"], 1)
        self.assertEqual(result["summary"]["paste_failures"], 0)
        self.assertEqual(len(result["results"]), 1)

    def test_keyword_ablation_runs_with_and_without_case_keywords(self):
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
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            engine = KeywordAblationEngine()
            result = run_keyword_ablation(manifest, engine)

        self.assertEqual(
            engine.calls,
            [
                ("warm.wav", True, ["PyTorch", "ROCm"]),
                ("warm.wav", True, []),
                ("measured.wav", True, ["PyTorch", "ROCm"]),
                ("measured.wav", True, []),
            ],
        )
        self.assertEqual(result["summary"]["warmup_cases"], 1)
        self.assertEqual(result["summary"]["cases"], 1)
        self.assertEqual(result["summary"]["keyword_cases"], 1)
        self.assertEqual(result["summary"]["mean_with_keywords_preservation"], 1.0)
        self.assertEqual(result["summary"]["mean_without_keywords_preservation"], 0.0)
        self.assertEqual(result["summary"]["mean_keyword_preservation_delta"], 1.0)
        self.assertEqual(result["summary"]["improved_cases"], 1)
        self.assertEqual(result["summary"]["regressed_cases"], 0)


if __name__ == "__main__":
    unittest.main()
