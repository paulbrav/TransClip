from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record one 16 kHz mono WAV plus matching reference text for real-usage eval.",
    )
    parser.add_argument("clip_dir", type=Path)
    parser.add_argument("stem", help="Clip filename stem, for example case_01 or warmup")
    parser.add_argument("--reference", help="Reference transcript text")
    parser.add_argument("--keywords", nargs="*", default=[], help="Optional per-clip keywords")
    parser.add_argument("--duration", type=int, default=10, help="Maximum recording duration in seconds")
    parser.add_argument("--device", default="default", help="arecord capture device")
    args = parser.parse_args(argv)

    if not shutil.which("arecord"):
        print("arecord is not installed", file=sys.stderr)
        return 1
    if args.duration <= 0:
        print("--duration must be positive", file=sys.stderr)
        return 1

    reference = args.reference if args.reference is not None else input("Reference: ").strip()
    if not reference.strip():
        print("reference text is required", file=sys.stderr)
        return 1

    args.clip_dir.mkdir(parents=True, exist_ok=True)
    wav_path = args.clip_dir / f"{args.stem}.wav"
    reference_path = args.clip_dir / f"{args.stem}.txt"
    keyword_path = args.clip_dir / f"{args.stem}.keywords.txt"

    command = arecord_command(wav_path, args.device, args.duration)
    print(f"recording {args.duration}s to {wav_path}")
    subprocess.run(command, check=True)
    reference_path.write_text(reference.strip() + "\n", encoding="utf-8")
    if args.keywords:
        keyword_path.write_text("\n".join(args.keywords) + "\n", encoding="utf-8")
    print(f"wrote {reference_path}")
    return 0


def arecord_command(wav_path: Path, device: str, duration: int) -> list[str]:
    return [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-d",
        str(duration),
        str(wav_path),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
