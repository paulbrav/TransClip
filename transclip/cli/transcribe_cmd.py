from __future__ import annotations

import argparse
import json
import sys

from transclip.service import InferenceEngine
from transclip.settings import Settings


def handle_transcribe(args: argparse.Namespace, settings: Settings) -> int:
    engine = InferenceEngine(settings)
    result = engine.transcribe(args.wav, cleanup=not args.no_cleanup)
    print(result["raw_asr"] if args.raw else result["text"])
    print(json.dumps({"timings_ms": result["timings_ms"]}), file=sys.stderr)
    return 0


def handle_cleanup(args: argparse.Namespace, settings: Settings) -> int:
    engine = InferenceEngine(settings)
    text = " ".join(args.text) if args.text else sys.stdin.read()
    print(engine.cleanup_text(text)["text"])
    return 0
