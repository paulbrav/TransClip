#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from granite_speach.eval_harness import run_keyword_ablation  # noqa: E402
from granite_speach.service import InferenceEngine  # noqa: E402
from granite_speach.settings import keywords_path, load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a glossary on/off keyword-preservation ablation for an eval manifest.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "real-usage" / "keyword-ablation.json")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings) if args.settings else load_settings()
    config_dir = args.settings.parent if args.settings else None
    engine = InferenceEngine(settings, keyword_path=keywords_path(config_dir))
    result = run_keyword_ablation(args.manifest, engine)
    encoded = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
