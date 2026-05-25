from __future__ import annotations

import argparse
import json
import sys

from transclip.notify import notify
from transclip.product import CLI_COMMAND, DISPLAY_NAME
from transclip.recording_ops import toggle_recording
from transclip.settings import Settings


def handle_toggle_record(args: argparse.Namespace, settings: Settings) -> int:
    outcome = toggle_recording(settings, paste=args.paste)
    if not outcome.ok:
        if outcome.error_message == "TransClip service is not running.":
            print(f"{outcome.error_message} Start it with: {CLI_COMMAND} serve", file=sys.stderr)
        else:
            print(outcome.error_message, file=sys.stderr)
        notify(DISPLAY_NAME, outcome.notification_message)
        return 1
    if outcome.paste_failed_message or outcome.paste_notice_message:
        notify(DISPLAY_NAME, outcome.notification_message)
    print(json.dumps(outcome.payload))
    return 0
