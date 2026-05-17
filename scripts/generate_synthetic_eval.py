from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval" / "v1-synthetic"
AUDIO_DIR = EVAL_DIR / "audio"
TEXT_DIR = EVAL_DIR / "text"
VOICE = Path.home() / ".local/share/narrator/piper/en_US-amy-medium.onnx"
VOICE_CONFIG = Path.home() / ".local/share/narrator/piper/en_US-amy-medium.onnx.json"

CASES = [
    (
        "warmup",
        "PyTorch on ROCm with gfx1151 is ready for the Granite speech benchmark.",
        ["PyTorch", "ROCm", "gfx1151", "Granite"],
    ),
    (
        "case_01",
        "Please check the Python tray icon after the service reports ready.",
        ["Python tray"],
    ),
    (
        "case_02",
        "Set the cleanup runtime to rule for the latency profile.",
        ["cleanup"],
    ),
    (
        "case_03",
        "Granite NAR should stay under seven hundred milliseconds after warm up.",
        ["Granite", "NAR"],
    ),
    (
        "case_04",
        "Add PyTorch, ROCm, and gfx1151 to the keyword glossary.",
        ["PyTorch", "ROCm", "gfx1151"],
    ),
    (
        "case_05",
        "The localhost Python inference service is listening on port eight seven six five.",
        ["Python"],
    ),
    (
        "case_06",
        "Use wtype on Wayland and xdotool on X eleven when paste injection is needed.",
        ["wtype", "Wayland", "xdotool"],
    ),
    (
        "case_07",
        "The macOS menu bar item may need accessibility permission for command V.",
        ["macOS"],
    ),
    (
        "case_08",
        "Do not add cloud fallback or transcript telemetry in version one.",
        ["telemetry"],
    ),
    (
        "case_09",
        "Keep the Gemma cleanup backend available for quality comparisons.",
        ["Gemma"],
    ),
    (
        "case_10",
        "Open the keyword glossary and verify Hugging Face is preserved exactly.",
        ["Hugging Face"],
    ),
    (
        "case_11",
        "Run the unit tests after changing the Python tray.",
        ["Python tray"],
    ),
    (
        "case_12",
        "The recent transcripts submenu should keep the last five snippets.",
        ["recent transcripts"],
    ),
    (
        "case_13",
        "Clipboard restoration should wait five hundred milliseconds before replacing text.",
        ["Clipboard"],
    ),
    (
        "case_14",
        "The app should hide the secondary status window instead of quitting.",
        ["status window"],
    ),
    (
        "case_15",
        "Use the TheRock nightly index for the Radeon eight zero six zero S.",
        ["TheRock", "Radeon"],
    ),
    (
        "case_16",
        "FlashAttention Triton AMD is required for the Granite NAR backend.",
        ["FlashAttention", "Triton", "AMD", "Granite", "NAR"],
    ),
    (
        "case_17",
        "The debug capture folder should include audio dot wav and timings dot json.",
        ["debug capture"],
    ),
    (
        "case_18",
        "Avoid rewriting llama.cpp as llama.cpp during cleanup.",
        ["llama.cpp"],
    ),
    (
        "case_19",
        "The service health endpoint should include the active hotkey and paste shortcut.",
        ["health endpoint"],
    ),
    (
        "case_20",
        "Do not treat the Dock as the primary home for this app.",
        ["Dock"],
    ),
    (
        "case_21",
        "Use a Linux panel app indicator and a macOS menu bar status item.",
        ["Linux", "macOS"],
    ),
    (
        "case_22",
        "The eval manifest includes warmup cases that are excluded from metrics.",
        ["eval manifest"],
    ),
    (
        "case_23",
        "Keyword preservation should improve recognition of Qwen, MLX, and Transformers.",
        ["Qwen", "MLX", "Transformers"],
    ),
    (
        "case_24",
        "Stop recording when the global shortcut is released.",
        ["global shortcut"],
    ),
    (
        "case_25",
        "Leave the transcript on the clipboard when paste simulation fails.",
        ["clipboard"],
    ),
]


def main() -> int:
    piper = shutil.which("piper")
    if not piper:
        print("piper is not installed", file=sys.stderr)
        return 1
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg is not installed", file=sys.stderr)
        return 1
    if not VOICE.exists() or not VOICE_CONFIG.exists():
        print(f"missing Piper voice: {VOICE}", file=sys.stderr)
        return 1

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    for case_id, reference, keywords in CASES:
        text_path = TEXT_DIR / f"{case_id}.txt"
        wav_path = AUDIO_DIR / f"{case_id}.wav"
        raw_wav_path = AUDIO_DIR / f".{case_id}.raw.wav"
        text_path.write_text(reference + "\n", encoding="utf-8")
        subprocess.run(
            [
                piper,
                "--model",
                str(VOICE),
                "--config",
                str(VOICE_CONFIG),
                "--input-file",
                str(text_path),
                "--output-file",
                str(raw_wav_path),
                "--sentence-silence",
                "0.1",
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(raw_wav_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ],
            check=True,
        )
        raw_wav_path.unlink(missing_ok=True)
        entry = {
            "audio_path": f"audio/{case_id}.wav",
            "reference": reference,
            "keywords": keywords,
            "cleanup": True,
        }
        if case_id == "warmup":
            warmup = entry
        else:
            manifest_cases.append(entry)

    manifest = {"warmup_cases": [warmup], "cases": manifest_cases}
    (EVAL_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest_cases)} cases to {EVAL_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
