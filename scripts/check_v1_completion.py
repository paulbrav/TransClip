#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_eval_results import check_results  # noqa: E402
from granite_speach.doctor import Check, run_checks  # noqa: E402
from granite_speach.settings import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate V1 completion on real eval and host readiness evidence.",
    )
    parser.add_argument(
        "--real-results",
        type=Path,
        default=ROOT / "eval" / "real-usage" / "results.json",
    )
    parser.add_argument(
        "--synthetic-results",
        type=Path,
        default=ROOT / "eval" / "v1-synthetic" / "results.json",
    )
    parser.add_argument("--skip-doctor", action="store_true")
    args = parser.parse_args(argv)

    doctor_checks = [] if args.skip_doctor else run_checks(load_settings())
    report = check_completion(
        real_results_path=args.real_results,
        synthetic_results_path=args.synthetic_results,
        doctor_checks=doctor_checks,
        doctor_checked=not args.skip_doctor,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


def check_completion(
    real_results_path: Path,
    synthetic_results_path: Path,
    doctor_checks: list[Check],
    doctor_checked: bool = True,
) -> dict[str, Any]:
    blockers: list[str] = []
    evidence: dict[str, Any] = {}

    if synthetic_results_path.exists():
        try:
            evidence["synthetic_eval"] = check_results(
                json.loads(synthetic_results_path.read_text(encoding="utf-8")),
                min_cases=25,
                max_cases=25,
            )
        except ValueError as exc:
            blockers.append(f"synthetic eval gate failed: {exc}")
    else:
        blockers.append(f"synthetic eval results missing: {synthetic_results_path}")

    if real_results_path.exists():
        try:
            evidence["real_usage_eval"] = check_results(
                json.loads(real_results_path.read_text(encoding="utf-8")),
                min_cases=20,
                max_cases=30,
            )
        except ValueError as exc:
            blockers.append(f"real-usage eval gate failed: {exc}")
    else:
        blockers.append(f"real-usage eval results missing: {real_results_path}")

    if doctor_checked:
        failed_checks = [check for check in doctor_checks if not check.ok]
        evidence["doctor"] = [
            {"name": check.name, "ok": check.ok, "detail": check.detail}
            for check in doctor_checks
        ]
        for check in failed_checks:
            blockers.append(f"doctor {check.name} failed: {check.detail}")
    else:
        evidence["doctor"] = "skipped"

    return {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "evidence": evidence,
    }


if __name__ == "__main__":
    raise SystemExit(main())
