from __future__ import annotations

import argparse
import json
import sys

from transclip.audio import recording_debug
from transclip.daemon import (
    collect_status,
    install_daemon,
    run_smoke_test,
    service_action,
    stream_logs,
    uninstall_daemon,
)
from transclip.settings import Settings

from .formatting import format_status, print_command_results


def handle_daemon_command(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "install":
        results = install_daemon(settings_path=args.settings)
        print_command_results(results)
        return 0 if all(result.ok for result in results) else 1
    if args.command == "uninstall":
        results = uninstall_daemon()
        print_command_results(results)
        return 0 if all(result.ok for result in results) else 1
    if args.command in {"start", "stop", "restart"}:
        result = service_action(args.command)
        print_command_results([result])
        return 0 if result.ok else 1
    if args.command == "status":
        status = collect_status(settings)
        print(json.dumps(status, indent=2) if args.json else format_status(status))
        return 0 if status["ready"] else 1
    if args.command == "logs":
        return stream_logs(follow=args.follow)
    if args.command == "smoke-test":
        if args.recording_debug:
            try:
                print(json.dumps(recording_debug(settings), indent=2))
                return 0
            except Exception as exc:
                print(f"recording debug failed: {exc}", file=sys.stderr)
                return 1
        results = run_smoke_test(settings, paste=args.paste)
        print_command_results(results)
        return 0 if all(result.ok for result in results) else 1
    raise ValueError(args.command)
