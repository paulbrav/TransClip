from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transclip.eval_harness import build_manifest, load_keyword_file

DEFAULT_OUTPUT = Path("eval/real-usage/manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a real-usage eval manifest from WAV files and reference text files.",
    )
    parser.add_argument("clip_dir", type=Path, help="Directory containing .wav and matching .txt files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--warmup-stem",
        help="Optional clip stem to put in warmup_cases instead of measured cases",
    )
    parser.add_argument(
        "--global-keywords",
        type=Path,
        help="Optional keyword file; matching terms are attached to each case",
    )
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument(
        "--allow-small",
        action="store_true",
        help="Allow fewer than --min-cases measured cases for smoke tests",
    )
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(
            args.clip_dir,
            output_path=args.output,
            warmup_stem=args.warmup_stem,
            global_keywords=load_keyword_file(args.global_keywords) if args.global_keywords else [],
            min_cases=args.min_cases,
            max_cases=args.max_cases,
            allow_small=args.allow_small,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(manifest.get('cases', []))} measured cases"
        f" and {len(manifest.get('warmup_cases', []))} warmup cases to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
