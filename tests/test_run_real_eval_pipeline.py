import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_real_eval_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_real_eval_pipeline", SCRIPT_PATH)
run_real_eval_pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(run_real_eval_pipeline)


def passing_results(case_count: int = 20) -> dict:
    results = [
        {
            "timings_ms": {"end_to_end": 250.0},
            "keyword_preservation": 1.0,
            "wer": 0.0,
        }
        for _ in range(case_count)
    ]
    return {
        "summary": {
            "cases": case_count,
            "mean_release_to_ready_ms": 250.0,
            "under_700ms": case_count,
            "mean_keyword_preservation": 1.0,
            "mean_wer": 0.0,
            "cleanup_semantic_drift_failures": 0,
            "paste_failures": 0,
        },
        "results": results,
    }


class RunRealEvalPipelineTests(unittest.TestCase):
    def test_missing_clips_fail_before_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = run_real_eval_pipeline.main(
                [
                    str(root / "missing"),
                    "--manifest",
                    str(root / "manifest.json"),
                    "--results",
                    str(root / "results.json"),
                    "--synthetic-results",
                    str(root / "synthetic.json"),
                    "--skip-doctor",
                ]
            )

        self.assertEqual(rc, 1)

    def test_pipeline_writes_manifest_runs_eval_and_checks_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clips = root / "clips"
            clips.mkdir()
            for index in range(20):
                stem = f"case_{index + 1:02d}"
                (clips / f"{stem}.wav").write_bytes(b"RIFF fake")
                (clips / f"{stem}.txt").write_text("Use PyTorch on ROCm.\n", encoding="utf-8")
            (clips / "warmup.wav").write_bytes(b"RIFF fake")
            (clips / "warmup.txt").write_text("Warm PyTorch on ROCm.\n", encoding="utf-8")
            keywords = root / "keywords.txt"
            keywords.write_text("PyTorch\nROCm\n", encoding="utf-8")
            manifest = root / "eval" / "manifest.json"
            results = root / "eval" / "results.json"
            synthetic = root / "eval" / "synthetic.json"
            synthetic.parent.mkdir()
            synthetic.write_text(json.dumps(passing_results(25)) + "\n", encoding="utf-8")
            commands = []

            def fake_run(command, **_kwargs):
                commands.append(command)
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(passing_results()) + "\n", encoding="utf-8")
                return type("Completed", (), {"returncode": 0})()

            with patch("run_real_eval_pipeline.subprocess.run", side_effect=fake_run):
                rc = run_real_eval_pipeline.main(
                    [
                        str(clips),
                        "--manifest",
                        str(manifest),
                        "--results",
                        str(results),
                        "--synthetic-results",
                        str(synthetic),
                        "--global-keywords",
                        str(keywords),
                        "--skip-doctor",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(commands[0][:3], [run_real_eval_pipeline.sys.executable, "-m", "transclip.cli"])
            written_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(written_manifest["cases"]), 20)
            self.assertEqual(len(written_manifest["warmup_cases"]), 1)
            self.assertEqual(written_manifest["cases"][0]["keywords"], ["PyTorch", "ROCm"])


if __name__ == "__main__":
    unittest.main()
