import unittest

from transclip.eval_harness import check_results


def payload(cases=20, latency=300.0, keyword=1.0, wer=0.1):
    return {
        "summary": {
            "cases": cases,
            "mean_release_to_ready_ms": latency,
            "under_700ms": cases,
            "under_1500ms": cases,
            "mean_keyword_preservation": keyword,
            "mean_wer": wer,
            "cleanup_semantic_drift_failures": 0,
            "paste_failures": 0,
        },
        "results": [
            {
                "audio_path": f"case_{index:02d}.wav",
                "text": "ok",
                "raw_asr": "ok",
                "reference": "ok",
                "wer": wer,
                "raw_asr_wer": wer,
                "cleanup_drift_wer_delta": 0.0,
                "cleanup_semantic_drift": False,
                "keyword_preservation": keyword,
                "paste_attempted": None,
                "paste_success": None,
                "timings_ms": {"end_to_end": latency},
            }
            for index in range(cases)
        ],
    }


class CheckEvalResultsTests(unittest.TestCase):
    def test_passes_v1_result_shape(self):
        report = check_results(payload())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["cases"], 20)

    def test_rejects_too_few_cases(self):
        with self.assertRaisesRegex(ValueError, "expected 20-30"):
            check_results(payload(cases=19))

    def test_rejects_latency_failures(self):
        with self.assertRaisesRegex(ValueError, "worst latency"):
            check_results(payload(latency=1600.0))

    def test_rejects_keyword_failures(self):
        with self.assertRaisesRegex(ValueError, "keyword preservation"):
            check_results(payload(keyword=0.5))

    def test_rejects_wer_failures(self):
        with self.assertRaisesRegex(ValueError, "mean WER"):
            check_results(payload(wer=0.5))

    def test_rejects_cleanup_drift_failures(self):
        bad = payload()
        bad["summary"]["cleanup_semantic_drift_failures"] = 1
        with self.assertRaisesRegex(ValueError, "cleanup semantic drift"):
            check_results(bad)

    def test_rejects_paste_failures(self):
        bad = payload()
        bad["summary"]["paste_failures"] = 1
        with self.assertRaisesRegex(ValueError, "paste failures"):
            check_results(bad)


if __name__ == "__main__":
    unittest.main()
