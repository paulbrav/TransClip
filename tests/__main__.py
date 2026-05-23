"""Run the test suite with one subprocess per module for clean interpreter teardown."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _test_modules() -> list[str]:
    root = Path(__file__).resolve().parent
    return sorted(
        path.stem.replace("/", ".")
        for path in root.glob("test_*.py")
    )


def _run_module(module: str) -> tuple[int, int, int]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", f"tests.{module}", "-v"],
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests?", output)
    count = int(match.group(1)) if match else 0
    if completed.returncode != 0:
        print(output, end="" if output.endswith("\n") else "\n", file=sys.stderr)
    return completed.returncode, count, count if completed.returncode == 0 else 0


def main() -> int:
    modules = _test_modules()
    total = passed_modules = 0
    failed: list[str] = []

    for module in modules:
        exit_code, count, _passed = _run_module(module)
        total += count
        if exit_code == 0:
            passed_modules += 1
            print(f"{module}: {count} ok")
        else:
            failed.append(module)
            print(f"{module}: FAILED (exit={exit_code})", file=sys.stderr)

    print(f"\nRan {total} tests in {len(modules)} modules")
    if failed:
        print(f"FAILED modules: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
