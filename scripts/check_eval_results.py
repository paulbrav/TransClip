from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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


def check_results(
    payload: dict[str, Any],
    min_cases: int = 20,
    max_cases: int = 30,
    max_mean_latency_ms: float = 700.0,
    max_latency_ms: float = 1500.0,
    min_under_700_ratio: float = 0.8,
    min_keyword_preservation: float = 0.9,
    max_mean_wer: float = 0.25,
    max_cleanup_drift_failures: int = 0,
    max_paste_failures: int = 0,
) -> dict[str, Any]:
    summary = payload.get("summary")
    results = payload.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError("results JSON must contain summary and results")

    cases = int(summary.get("cases", len(results)))
    if cases != len(results):
        raise ValueError(f"summary cases {cases} does not match results length {len(results)}")
    if cases < min_cases or cases > max_cases:
        raise ValueError(f"expected {min_cases}-{max_cases} measured cases, found {cases}")

    latencies = [float(result.get("timings_ms", {}).get("end_to_end", 0.0)) for result in results]
    if not latencies or any(value <= 0 for value in latencies):
        raise ValueError("each result must include a positive timings_ms.end_to_end")
    worst_latency = max(latencies)
    if worst_latency > max_latency_ms:
        raise ValueError(f"worst latency {worst_latency:.3f}ms exceeds {max_latency_ms:.3f}ms")

    mean_latency = float(summary.get("mean_release_to_ready_ms") or 0.0)
    if mean_latency <= 0:
        raise ValueError("summary.mean_release_to_ready_ms must be positive")
    if mean_latency > max_mean_latency_ms:
        raise ValueError(f"mean latency {mean_latency:.3f}ms exceeds {max_mean_latency_ms:.3f}ms")

    under_700 = int(summary.get("under_700ms", 0))
    under_700_ratio = under_700 / cases
    if under_700_ratio < min_under_700_ratio:
        raise ValueError(f"under-700ms ratio {under_700_ratio:.3f} is below {min_under_700_ratio:.3f}")

    keyword_preservation = summary.get("mean_keyword_preservation")
    if keyword_preservation is not None and float(keyword_preservation) < min_keyword_preservation:
        raise ValueError(
            f"mean keyword preservation {float(keyword_preservation):.3f} is below {min_keyword_preservation:.3f}"
        )

    mean_wer = summary.get("mean_wer")
    if mean_wer is not None and float(mean_wer) > max_mean_wer:
        raise ValueError(f"mean WER {float(mean_wer):.3f} exceeds {max_mean_wer:.3f}")

    cleanup_drift_failures = int(summary.get("cleanup_semantic_drift_failures", 0))
    if cleanup_drift_failures > max_cleanup_drift_failures:
        raise ValueError(
            f"cleanup semantic drift failures {cleanup_drift_failures} exceed {max_cleanup_drift_failures}"
        )

    paste_failures = int(summary.get("paste_failures", 0))
    if paste_failures > max_paste_failures:
        raise ValueError(f"paste failures {paste_failures} exceed {max_paste_failures}")

    return {
        "status": "pass",
        "cases": cases,
        "mean_release_to_ready_ms": mean_latency,
        "worst_release_to_ready_ms": worst_latency,
        "under_700_ratio": under_700_ratio,
        "mean_keyword_preservation": keyword_preservation,
        "mean_wer": mean_wer,
        "cleanup_semantic_drift_failures": cleanup_drift_failures,
        "paste_failures": paste_failures,
    }


if __name__ == "__main__":
    raise SystemExit(main())
