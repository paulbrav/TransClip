from __future__ import annotations

import argparse
from pathlib import Path

from .cli_commands import handle_command
from .product import CLI_COMMAND


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=CLI_COMMAND)
    parser.add_argument("--settings", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-config")
    sub.add_parser("serve")
    sub.add_parser("install")
    sub.add_parser("uninstall")
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("restart")
    tray = sub.add_parser("tray")
    tray.add_argument("--no-system-python-fallback", action="store_true", help=argparse.SUPPRESS)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--json", action="store_true")

    logs_parser = sub.add_parser("logs")
    logs_parser.add_argument("-f", "--follow", action="store_true")

    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--paste", action="store_true")
    smoke.add_argument("--recording-debug", action="store_true")

    history = sub.add_parser("history")
    history.add_argument("--limit", type=int, default=10)
    history.add_argument("--json", action="store_true")
    history.add_argument("--copy", type=int)

    models = sub.add_parser("models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list")
    models_sub.add_parser("doctor")
    models_prefetch = models_sub.add_parser("prefetch")
    models_prefetch.add_argument("--model", required=True)

    config = sub.add_parser("config")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_get = config_sub.add_parser("get")
    config_get.add_argument("field", nargs="?")
    config_set = config_sub.add_parser("set")
    config_set.add_argument("field")
    config_set.add_argument("value")

    transcribe = sub.add_parser("transcribe")
    transcribe.add_argument("wav", type=Path)
    transcribe.add_argument("--raw", action="store_true")
    transcribe.add_argument("--no-cleanup", action="store_true")

    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("text", nargs="*")

    record = sub.add_parser("record-once")
    record.add_argument("--seconds", type=float, default=5.0)
    record.add_argument("--paste", action="store_true")

    toggle = sub.add_parser("toggle-record")
    toggle.add_argument("--paste", action="store_true")

    gnome_shortcut = sub.add_parser("install-gnome-shortcut")
    gnome_shortcut.add_argument("--binding")
    gnome_shortcut.add_argument("--command", dest="shortcut_command")

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("manifest", type=Path)
    eval_parser.add_argument("--output", type=Path)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--fix", action="store_true")
    doctor.add_argument("--audio-debug", action="store_true")

    args = parser.parse_args(argv)
    return handle_command(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
