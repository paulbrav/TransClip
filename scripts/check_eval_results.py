from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transclip.eval_harness import check_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check eval results against V1 pass criteria.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--max-mean-latency-ms", type=float, default=700.0)
    parser.add_argument("--max-latency-ms", type=float, default=1500.0)
    parser.add_argument("--min-under-700-ratio", type=float, default=0.8)
    parser.add_argument("--min-keyword-preservation", type=float, default=0.9)
    parser.add_argument("--max-mean-wer", type=float, default=0.25)
    parser.add_argument("--max-cleanup-drift-failures", type=int, default=0)
    parser.add_argument("--max-paste-failures", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        report = check_results(
            json.loads(args.results.read_text(encoding="utf-8")),
            min_cases=args.min_cases,
            max_cases=args.max_cases,
            max_mean_latency_ms=args.max_mean_latency_ms,
            max_latency_ms=args.max_latency_ms,
            min_under_700_ratio=args.min_under_700_ratio,
            min_keyword_preservation=args.min_keyword_preservation,
            max_mean_wer=args.max_mean_wer,
            max_cleanup_drift_failures=args.max_cleanup_drift_failures,
            max_paste_failures=args.max_paste_failures,
        )
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
