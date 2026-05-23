from __future__ import annotations

import argparse
import sys

from transclip.doctor import checks_as_text, run_model_checks
from transclip.models import model_rows, prefetch_model
from transclip.settings import Settings

from .formatting import format_model_rows


def handle_models(args: argparse.Namespace, settings: Settings) -> int:
    if args.models_command == "list":
        print(format_model_rows(model_rows(settings)))
        return 0
    if args.models_command == "doctor":
        checks = run_model_checks(settings)
        print(checks_as_text(checks))
        return 0 if all(check.ok for check in checks) else 1
    if args.models_command == "prefetch":
        try:
            path = prefetch_model(args.model, settings)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"ok\tprefetched {args.model} under {path}")
        return 0
    raise ValueError(args.models_command)
