from __future__ import annotations

import argparse

from transclip.settings import write_default_settings


def handle_init_config(args: argparse.Namespace) -> int:
    print(write_default_settings(args.settings))
    return 0
