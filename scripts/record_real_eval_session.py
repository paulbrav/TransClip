from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from generate_synthetic_eval import CASES  # noqa: E402
from record_real_eval_clip import arecord_command  # noqa: E402


class Recorder(Protocol):
    def start(self) -> None: ...

    def stop_to_wav(self, output_path: Path) -> Path: ...


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactively record the standard 25-case V1 real-usage eval set.",
    )
    parser.add_argument("clip_dir", type=Path)
    parser.add_argument("--duration", type=int, default=10, help="Maximum seconds per clip")
    parser.add_argument("--device", default="default", help="arecord capture device")
    parser.add_argument(
        "--manual-stop",
        action="store_true",
        help="Record each clip until Enter is pressed again, using the app microphone recorder",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="Settings TOML to use with --manual-stop, defaulting to the app config",
    )
    parser.add_argument("--start-at", default="warmup", help="First case id to record")
    parser.add_argument("--limit", type=int, help="Maximum number of cases to record")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the warmup clip")
    parser.add_argument("--yes", action="store_true", help="Record without prompting before each clip")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clip files")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned session without recording")
    parser.add_argument(
        "--prompt-sheet",
        type=Path,
        help="Write the selected prompts to a Markdown file and exit",
    )
    args = parser.parse_args(argv)

    if args.duration <= 0:
        print("--duration must be positive", file=sys.stderr)
        return 1
    if not args.dry_run and not args.manual_stop and not shutil.which("arecord"):
        print("arecord is not installed", file=sys.stderr)
        return 1

    try:
        selected = select_cases(
            start_at=args.start_at,
            limit=args.limit,
            include_warmup=not args.no_warmup,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.prompt_sheet:
        args.prompt_sheet.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_sheet.write_text(prompt_sheet(selected), encoding="utf-8")
        print(f"wrote prompt sheet to {args.prompt_sheet}")
        return 0

    args.clip_dir.mkdir(parents=True, exist_ok=True)
    recorded = 0
    for index, (case_id, reference, keywords) in enumerate(selected, start=1):
        wav_path = args.clip_dir / f"{case_id}.wav"
        print(f"\n[{index}/{len(selected)}] {case_id}")
        print(reference)
        if keywords:
            print("keywords: " + ", ".join(keywords))

        if args.dry_run:
            continue
        if wav_path.exists() and not args.overwrite:
            print(f"skipping existing {wav_path}; pass --overwrite to replace it")
            continue
        if not args.yes:
            response = input("Press Enter to record, s to skip, q to quit: ").strip().lower()
            if response == "q":
                break
            if response == "s":
                continue

        if args.manual_stop:
            record_manual_clip(wav_path, args.settings)
        else:
            subprocess.run(arecord_command(wav_path, args.device, args.duration), check=True)
        write_case_metadata(args.clip_dir, case_id, reference, keywords)
        recorded += 1

    measured = count_measured_cases(args.clip_dir)
    print(f"\nrecorded {recorded} clips; found {measured} measured case clips in {args.clip_dir}")
    print(
        "next: uv run scripts/prepare_real_eval.py "
        f"{args.clip_dir} --warmup-stem warmup --global-keywords ~/.config/granite-speach/keywords.txt "
        "--output eval/real-usage/manifest.json"
    )
    return 0


def select_cases(
    start_at: str = "warmup",
    limit: int | None = None,
    include_warmup: bool = True,
) -> list[tuple[str, str, list[str]]]:
    cases = [case for case in CASES if include_warmup or case[0] != "warmup"]
    ids = [case[0] for case in cases]
    if start_at not in ids:
        raise ValueError(f"unknown start case {start_at}; expected one of: {', '.join(ids)}")
    selected = cases[ids.index(start_at) :]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def write_case_metadata(
    clip_dir: Path,
    case_id: str,
    reference: str,
    keywords: list[str],
) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / f"{case_id}.txt").write_text(reference.strip() + "\n", encoding="utf-8")
    keyword_path = clip_dir / f"{case_id}.keywords.txt"
    if keywords:
        keyword_path.write_text("\n".join(keywords) + "\n", encoding="utf-8")
    elif keyword_path.exists():
        keyword_path.unlink()


def count_measured_cases(clip_dir: Path) -> int:
    return len([path for path in clip_dir.glob("case_*.wav") if path.with_suffix(".txt").exists()])


def record_manual_clip(
    wav_path: Path,
    settings_path: Path | None = None,
    input_fn: Callable[[str], str] = input,
    recorder: Recorder | None = None,
) -> Path:
    if recorder is None:
        from granite_speach.audio import AudioRecorder
        from granite_speach.settings import load_settings

        recorder = AudioRecorder(load_settings(settings_path))
    recorder.start()
    try:
        input_fn("Recording... press Enter to stop: ")
    finally:
        recorder.stop_to_wav(wav_path)
    return wav_path


def prompt_sheet(cases: list[tuple[str, str, list[str]]]) -> str:
    lines = [
        "# Granite Speach V1 Real-Usage Eval Prompts",
        "",
        "Read each prompt aloud when recording the matching case.",
        "",
    ]
    for index, (case_id, reference, keywords) in enumerate(cases, start=1):
        lines.append(f"## {index}. {case_id}")
        lines.append("")
        lines.append(reference)
        if keywords:
            lines.append("")
            lines.append("Keywords: " + ", ".join(keywords))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
