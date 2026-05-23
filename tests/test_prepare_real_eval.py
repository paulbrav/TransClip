import importlib.util
import tempfile
import unittest
from pathlib import Path

from transclip.eval_harness import build_manifest, load_keyword_file

RECORD_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_real_eval_clip.py"
RECORD_SPEC = importlib.util.spec_from_file_location("record_real_eval_clip", RECORD_SCRIPT_PATH)
record_real_eval_clip = importlib.util.module_from_spec(RECORD_SPEC)
assert RECORD_SPEC and RECORD_SPEC.loader
RECORD_SPEC.loader.exec_module(record_real_eval_clip)

SESSION_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_real_eval_session.py"
SESSION_SPEC = importlib.util.spec_from_file_location("record_real_eval_session", SESSION_SCRIPT_PATH)
record_real_eval_session = importlib.util.module_from_spec(SESSION_SPEC)
assert SESSION_SPEC and SESSION_SPEC.loader
SESSION_SPEC.loader.exec_module(record_real_eval_session)


class PrepareRealEvalTests(unittest.TestCase):
    def test_build_manifest_pairs_wavs_references_and_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clips = root / "clips"
            clips.mkdir()
            output = root / "eval" / "manifest.json"
            global_keywords = root / "keywords.txt"
            global_keywords.write_text("PyTorch\nROCm\nunused\n", encoding="utf-8")
            for stem, text in {
                "warmup": "Warm PyTorch on ROCm.",
                "case_01": "Use PyTorch on ROCm.",
                "case_02": "Plain dictation.",
            }.items():
                (clips / f"{stem}.wav").write_bytes(b"not really wav")
                (clips / f"{stem}.txt").write_text(text, encoding="utf-8")
            (clips / "case_02.keywords.txt").write_text("Wayland\n", encoding="utf-8")

            manifest = build_manifest(
                clips,
                output_path=output,
                warmup_stem="warmup",
                global_keywords=load_keyword_file(global_keywords),
                allow_small=True,
            )

            self.assertEqual(len(manifest["warmup_cases"]), 1)
            self.assertEqual(len(manifest["cases"]), 2)
            self.assertEqual(manifest["cases"][0]["keywords"], ["PyTorch", "ROCm"])
            self.assertEqual(manifest["cases"][1]["keywords"], ["Wayland"])
            self.assertEqual(manifest["cases"][0]["audio_path"], "../clips/case_01.wav")

    def test_requires_real_case_count_without_allow_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = Path(tmp) / "clips"
            clips.mkdir()
            (clips / "case.wav").write_bytes(b"not really wav")
            (clips / "case.txt").write_text("hello", encoding="utf-8")

            output = clips.parent / "manifest.json"
            with self.assertRaisesRegex(ValueError, "expected at least 20"):
                build_manifest(clips, output_path=output)

    def test_missing_reference_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = Path(tmp) / "clips"
            clips.mkdir()
            (clips / "case.wav").write_bytes(b"not really wav")

            with self.assertRaisesRegex(ValueError, "missing reference text"):
                build_manifest(clips, allow_small=True)

    def test_record_clip_arecord_command(self):
        command = record_real_eval_clip.arecord_command(
            Path("clips/case_01.wav"),
            "default",
            7,
        )

        self.assertEqual(
            command,
            [
                "arecord",
                "-D",
                "default",
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-d",
                "7",
                "clips/case_01.wav",
            ],
        )

    def test_real_eval_session_selects_cases_and_writes_metadata(self):
        selected = record_real_eval_session.select_cases(
            start_at="case_02",
            limit=2,
            include_warmup=False,
        )

        self.assertEqual([case[0] for case in selected], ["case_02", "case_03"])
        with tempfile.TemporaryDirectory() as tmp:
            clips = Path(tmp)
            record_real_eval_session.write_case_metadata(
                clips,
                "case_02",
                "Set the cleanup runtime to rule for the latency profile.",
                ["cleanup"],
            )

            self.assertEqual(
                (clips / "case_02.txt").read_text(encoding="utf-8"),
                "Set the cleanup runtime to rule for the latency profile.\n",
            )
            self.assertEqual(
                (clips / "case_02.keywords.txt").read_text(encoding="utf-8"),
                "cleanup\n",
            )

    def test_real_eval_session_writes_prompt_sheet(self):
        selected = record_real_eval_session.select_cases(
            start_at="case_01",
            limit=1,
            include_warmup=False,
        )

        sheet = record_real_eval_session.prompt_sheet(selected)

        self.assertIn("# TransClip V1 Real-Usage Eval Prompts", sheet)
        self.assertIn("## 1. case_01", sheet)
        self.assertIn("Please check the Python tray icon", sheet)
        self.assertIn("Keywords: Python tray", sheet)

    def test_real_eval_session_manual_record_starts_and_stops_recorder(self):
        class FakeRecorder:
            def __init__(self):
                self.started = False
                self.output_path = None

            def start(self):
                self.started = True

            def stop_to_wav(self, output_path):
                self.output_path = output_path
                output_path.write_bytes(b"wav")
                return output_path

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "case_01.wav"
            recorder = FakeRecorder()
            prompts = []

            result = record_real_eval_session.record_manual_clip(
                wav_path,
                input_fn=lambda prompt: prompts.append(prompt) or "",
                recorder=recorder,
            )

            self.assertEqual(result, wav_path)
            self.assertTrue(recorder.started)
            self.assertEqual(recorder.output_path, wav_path)
            self.assertEqual(wav_path.read_bytes(), b"wav")
            self.assertEqual(prompts, ["Recording... press Enter to stop: "])


if __name__ == "__main__":
    unittest.main()
