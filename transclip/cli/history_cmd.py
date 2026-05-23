from __future__ import annotations

import argparse
import json

from transclip.history import read_history

from .formatting import copy_history, format_history


def handle_history(args: argparse.Namespace) -> int:
    # History always reads the canonical user log dir; --settings does not relocate it.
    events = read_history(limit=args.limit)
    if args.copy is not None:
        return copy_history(events, args.copy)
    if args.json:
        print(json.dumps(events, indent=2))
    else:
        print(format_history(events))
    return 0
