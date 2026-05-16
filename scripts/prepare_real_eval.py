from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


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


def build_manifest(
    clip_dir: Path,
    output_path: Path = DEFAULT_OUTPUT,
    warmup_stem: str | None = None,
    global_keywords: list[str] | None = None,
    min_cases: int = 20,
    max_cases: int = 30,
    allow_small: bool = False,
) -> dict[str, list[dict[str, object]]]:
    clip_dir = clip_dir.expanduser().resolve()
    if not clip_dir.is_dir():
        raise ValueError(f"clip directory does not exist: {clip_dir}")

    global_keywords = global_keywords or []
    entries = []
    for wav_path in sorted(clip_dir.glob("*.wav")):
        reference_path = wav_path.with_suffix(".txt")
        if not reference_path.exists():
            raise ValueError(f"missing reference text for {wav_path.name}: {reference_path.name}")
        reference = reference_path.read_text(encoding="utf-8").strip()
        if not reference:
            raise ValueError(f"empty reference text: {reference_path}")
        keywords = keywords_for_case(wav_path, reference, global_keywords)
        entries.append(
            {
                "audio_path": manifest_path(wav_path, output_path),
                "reference": reference,
                "keywords": keywords,
                "cleanup": True,
            }
        )

    if not entries:
        raise ValueError(f"no .wav clips found in {clip_dir}")

    warmup_cases = []
    measured = entries
    if warmup_stem:
        warmup_name = f"{warmup_stem}.wav"
        warmup_cases = [entry for entry in entries if Path(str(entry["audio_path"])).name == warmup_name]
        if not warmup_cases:
            raise ValueError(f"warmup stem not found: {warmup_stem}")
        measured = [entry for entry in entries if Path(str(entry["audio_path"])).name != warmup_name]

    if len(measured) > max_cases:
        raise ValueError(f"expected at most {max_cases} measured clips, found {len(measured)}")
    if len(measured) < min_cases and not allow_small:
        raise ValueError(
            f"expected at least {min_cases} measured clips, found {len(measured)}; "
            "pass --allow-small only for smoke tests"
        )

    manifest: dict[str, list[dict[str, object]]] = {"cases": measured}
    if warmup_cases:
        manifest = {"warmup_cases": warmup_cases, "cases": measured}
    return manifest


def keywords_for_case(wav_path: Path, reference: str, global_keywords: list[str]) -> list[str]:
    keywords = []
    reference_lower = reference.lower()
    for keyword in global_keywords:
        if keyword.lower() in reference_lower:
            keywords.append(keyword)
    for suffix in (".keywords.txt", ".keywords"):
        keyword_path = wav_path.with_suffix(suffix)
        if keyword_path.exists():
            keywords.extend(load_keyword_file(keyword_path))
            break
    return list(dict.fromkeys(keywords))


def load_keyword_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def manifest_path(path: Path, output_path: Path) -> str:
    output_dir = output_path.expanduser().resolve().parent
    return os.path.relpath(path.resolve(), output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
