from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

from .audio import AudioRecorder
from .doctor import checks_as_json, checks_as_text, run_checks
from .eval_harness import run_eval
from .notify import notify
from .paste import paste_transcript
from .service import InferenceEngine, run_server
from .settings import (
    keywords_path,
    load_settings,
    write_default_keywords,
    write_default_settings,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="granite-speach")
    parser.add_argument("--settings", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-config")
    sub.add_parser("serve")

    transcribe = sub.add_parser("transcribe")
    transcribe.add_argument("wav", type=Path)
    transcribe.add_argument("--raw", action="store_true")
    transcribe.add_argument("--no-cleanup", action="store_true")

    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("text", nargs="*")

    record = sub.add_parser("record-once")
    record.add_argument("--seconds", type=float, default=5.0)
    record.add_argument("--paste", action="store_true")

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("manifest", type=Path)
    eval_parser.add_argument("--output", type=Path)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    config_dir = args.settings.parent if args.settings else None
    keyword_path = keywords_path(config_dir)
    if args.command == "init-config":
        print(write_default_settings(args.settings))
        print(write_default_keywords(keyword_path))
        return 0

    settings = load_settings(args.settings) if args.settings else load_settings()
    if args.command == "serve":
        run_server(settings, keyword_path=keyword_path)
        return 0

    if args.command == "doctor":
        checks = run_checks(settings, config_dir=config_dir)
        print(checks_as_json(checks) if args.json else checks_as_text(checks))
        return 0 if all(check.ok for check in checks) else 1

    engine = InferenceEngine(settings, keyword_path=keyword_path)
    if args.command == "transcribe":
        result = engine.transcribe(args.wav, cleanup=not args.no_cleanup)
        print(result["raw_asr"] if args.raw else result["text"])
        print(json.dumps({"timings_ms": result["timings_ms"]}), file=sys.stderr)
        return 0

    if args.command == "cleanup":
        text = " ".join(args.text) if args.text else sys.stdin.read()
        print(engine.cleanup_text(text)["text"])
        return 0

    if args.command == "record-once":
        recorder = AudioRecorder(settings)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "recording.wav"
            recorder.start()
            time.sleep(min(args.seconds, settings.max_recording_seconds))
            recorder.stop_to_wav(wav)
            result = engine.transcribe(wav)
            print(result["text"])
            if args.paste:
                paste_result = paste_transcript(result["text"], settings)
                if not paste_result.pasted:
                    detail = (
                        f" {paste_result.error_detail}"
                        if paste_result.error_detail
                        else ""
                    )
                    notify(
                        "Granite Speach",
                        "Paste failed. The transcript is still on the clipboard." + detail,
                    )
        return 0

    if args.command == "eval":
        result = run_eval(args.manifest, engine)
        encoded = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(encoded + "\n", encoding="utf-8")
        else:
            print(encoded)
        return 0

    parser.error(f"Unhandled command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
