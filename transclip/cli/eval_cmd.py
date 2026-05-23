from __future__ import annotations

import argparse
import json

from transclip.eval_harness import run_eval
from transclip.service import InferenceEngine
from transclip.settings import Settings


def handle_eval(args: argparse.Namespace, settings: Settings) -> int:
    engine = InferenceEngine(settings)
    result = run_eval(args.manifest, engine)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0
