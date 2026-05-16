#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from check_eval_results import check_results  # noqa: E402
from check_v1_completion import check_completion  # noqa: E402
from granite_speach.doctor import run_checks  # noqa: E402
from granite_speach.settings import default_config_dir, load_settings  # noqa: E402
from prepare_real_eval import build_manifest, load_keyword_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build, run, and gate the required V1 real-usage eval.",
    )
    parser.add_argument(
        "clip_dir",
        type=Path,
        nargs="?",
        default=Path("~/granite-real-eval"),
        help="Directory containing recorded WAV/reference pairs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "eval" / "real-usage" / "manifest.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "eval" / "real-usage" / "results.json",
    )
    parser.add_argument(
        "--synthetic-results",
        type=Path,
        default=ROOT / "eval" / "v1-synthetic" / "results.json",
    )
    parser.add_argument("--warmup-stem", default="warmup")
    parser.add_argument(
        "--global-keywords",
        type=Path,
        default=default_config_dir() / "keywords.txt",
    )
    parser.add_argument("--skip-doctor", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(
            args.clip_dir,
            output_path=args.manifest,
            warmup_stem=args.warmup_stem,
            global_keywords=load_keyword_file(args.global_keywords)
            if args.global_keywords.exists()
            else [],
        )
    except ValueError as exc:
        print(f"real eval clips are not ready: {exc}", file=sys.stderr)
        print(
            "record them with: uv run scripts/record_real_eval_session.py "
            f"{args.clip_dir.expanduser()} --manual-stop",
            file=sys.stderr,
        )
        return 1

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(manifest.get('cases', []))} measured cases"
        f" and {len(manifest.get('warmup_cases', []))} warmup cases to {args.manifest}"
    )

    eval_command = [
        sys.executable,
        "-m",
        "granite_speach.cli",
        "eval",
        str(args.manifest),
        "--output",
        str(args.results),
    ]
    print("+ " + " ".join(eval_command))
    completed = subprocess.run(
        eval_command,
        cwd=ROOT,
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode != 0:
        return completed.returncode

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    try:
        eval_report = check_results(payload)
    except ValueError as exc:
        print(f"real eval gate failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"real_usage_eval": eval_report}, indent=2))

    doctor_checks = [] if args.skip_doctor else run_checks(load_settings())
    completion_report = check_completion(
        real_results_path=args.results,
        synthetic_results_path=args.synthetic_results,
        doctor_checks=doctor_checks,
        doctor_checked=not args.skip_doctor,
    )
    print(json.dumps({"v1_completion": completion_report}, indent=2))
    return 0 if completion_report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
