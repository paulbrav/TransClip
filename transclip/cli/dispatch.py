from __future__ import annotations

import argparse

from transclip.settings import load_settings

from .config_cmd import handle_config
from .daemon_cmd import handle_daemon_command
from .doctor_cmd import handle_doctor
from .eval_cmd import handle_eval
from .history_cmd import handle_history
from .init_config import handle_init_config
from .models_cmd import handle_models
from .serve import handle_serve
from .shortcut_cmd import handle_gnome_shortcut
from .toggle_cmd import handle_toggle_record
from .transcribe_cmd import handle_cleanup, handle_transcribe
from .tray_cmd import handle_tray


def handle_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config_dir = args.settings.parent if args.settings else None
    if args.command == "init-config":
        return handle_init_config(args)

    settings = load_settings(args.settings) if args.settings else load_settings()
    if args.command == "serve":
        return handle_serve(settings)
    if args.command == "doctor":
        return handle_doctor(args, settings, config_dir)
    if args.command in {"install", "uninstall", "start", "stop", "restart", "status", "logs", "smoke-test"}:
        return handle_daemon_command(args, settings)
    if args.command == "tray":
        return handle_tray(settings, args.settings)
    if args.command == "history":
        return handle_history(args)
    if args.command == "models":
        return handle_models(args, settings)
    if args.command == "config":
        return handle_config(args, settings)
    if args.command == "install-gnome-shortcut":
        return handle_gnome_shortcut(args, settings)
    if args.command == "toggle-record":
        return handle_toggle_record(args, settings)
    if args.command == "transcribe":
        return handle_transcribe(args, settings)
    if args.command == "cleanup":
        return handle_cleanup(args, settings)
    if args.command == "eval":
        return handle_eval(args, settings)

    parser.error(f"Unhandled command {args.command}")
    return 2
