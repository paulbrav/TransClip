from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audio import recording_debug
from .daemon import (
    collect_status,
    logs_dir,
    run_smoke_test,
    stream_logs,
    toggle_log_path,
)
from .daemon_lifecycle import install_daemon, service_action, uninstall_daemon
from .doctor import (
    check_asr_runtime,
    check_model_cache,
    check_torch_runtime,
    checks_as_json,
    checks_as_text,
    run_checks,
)
from .eval_harness import run_eval
from .gnome_shortcut import install_shortcut
from .history import read_history
from .models import ModelRow, model_rows, prefetch_model
from .notify import notify
from .paste import SystemClipboard
from .platform_runtime import get_runtime
from .product import CLI_COMMAND, DISPLAY_NAME
from .recording_ops import toggle_recording
from .service import InferenceEngine, run_server
from .settings import (
    get_setting,
    load_settings,
    set_setting,
    settings_field_names,
    write_default_settings,
)


def handle_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config_dir = args.settings.parent if args.settings else None
    if args.command == "init-config":
        print(write_default_settings(args.settings))
        return 0

    settings = load_settings(args.settings) if args.settings else load_settings()
    if args.command == "serve":
        run_server(settings)
        return 0

    if args.command == "doctor":
        return handle_doctor(args, settings, config_dir)
    if args.command in {"install", "uninstall", "start", "stop", "restart", "status", "logs", "smoke-test"}:
        return handle_daemon_command(args, settings)
    if args.command == "tray":
        from .tray import run_tray

        return run_tray(settings, explicit_settings_path=args.settings)
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

    engine = InferenceEngine(settings)
    if args.command == "transcribe":
        result = engine.transcribe(args.wav, cleanup=not args.no_cleanup)
        print(result["raw_asr"] if args.raw else result["text"])
        print(json.dumps({"timings_ms": result["timings_ms"]}), file=sys.stderr)
        return 0
    if args.command == "cleanup":
        text = " ".join(args.text) if args.text else sys.stdin.read()
        print(engine.cleanup_text(text)["text"])
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


def handle_doctor(args: argparse.Namespace, settings, config_dir: Path | None) -> int:
    if args.fix:
        logs_dir().mkdir(parents=True, exist_ok=True)
        if get_runtime().system() == "Linux":
            try:
                install_shortcut(settings_path=args.settings, binding=settings.hotkey_linux)
            except Exception as exc:
                print(f"doctor --fix could not install GNOME shortcut: {exc}", file=sys.stderr)
    checks = run_checks(
        settings,
        config_dir=config_dir,
        include_audio_debug=args.audio_debug,
    )
    print(checks_as_json(checks) if args.json else checks_as_text(checks))
    return 0 if all(check.ok for check in checks) else 1


def handle_daemon_command(args: argparse.Namespace, settings) -> int:
    if args.command == "install":
        results = install_daemon(settings_path=args.settings)
        _print_command_results(results)
        return 0 if all(result.ok for result in results) else 1
    if args.command == "uninstall":
        results = uninstall_daemon()
        _print_command_results(results)
        return 0 if all(result.ok for result in results) else 1
    if args.command in {"start", "stop", "restart"}:
        result = service_action(args.command)
        _print_command_results([result])
        return 0 if result.ok else 1
    if args.command == "status":
        status = collect_status(settings)
        print(json.dumps(status, indent=2) if args.json else _format_status(status))
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
        _print_command_results(results)
        return 0 if all(result.ok for result in results) else 1
    raise ValueError(args.command)


def handle_history(args: argparse.Namespace) -> int:
    events = read_history(limit=args.limit)
    if args.copy is not None:
        return _copy_history(events, args.copy)
    if args.json:
        print(json.dumps(events, indent=2))
    else:
        print(_format_history(events))
    return 0


def handle_models(args: argparse.Namespace, settings) -> int:
    if args.models_command == "list":
        print(_format_model_rows(model_rows(settings)))
        return 0
    if args.models_command == "doctor":
        checks = [
            check_model_cache(settings),
            check_torch_runtime(settings),
            check_asr_runtime(settings),
        ]
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


def handle_config(args: argparse.Namespace, settings) -> int:
    try:
        if args.config_command == "get":
            if args.field:
                print(get_setting(settings, args.field))
            else:
                for field_name in settings_field_names():
                    print(f"{field_name} = {get_setting(settings, field_name)}")
            return 0
        if args.config_command == "set":
            set_setting(args.settings, args.field, args.value)
            print(f"set\t{args.field} = {args.value}")
            if args.field == "hotkey_linux":
                print(f"run: {CLI_COMMAND} install-gnome-shortcut or {CLI_COMMAND} doctor --fix")
            if args.field == "hotkey_windows":
                print(f"restart {CLI_COMMAND} tray to apply the new Windows hotkey")
            return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    raise ValueError(args.config_command)


def handle_gnome_shortcut(args: argparse.Namespace, settings) -> int:
    result = install_shortcut(
        settings_path=args.settings,
        command=args.shortcut_command,
        binding=args.binding or settings.hotkey_linux,
    )
    print(f"Installed {result.name}")
    print(f"Path: {result.path}")
    print(f"Binding: {result.binding}")
    print(f"Command: {result.command}")
    return 0


def handle_toggle_record(args: argparse.Namespace, settings) -> int:
    outcome = toggle_recording(settings, paste=args.paste)
    if not outcome.ok:
        if outcome.error_message == "TransClip service is not running.":
            print(f"{outcome.error_message} Start it with: {CLI_COMMAND} serve", file=sys.stderr)
        else:
            print(outcome.error_message, file=sys.stderr)
        notify(DISPLAY_NAME, outcome.notification_message)
        return 1
    if outcome.paste_failed_message:
        notify(DISPLAY_NAME, outcome.notification_message)
    print(json.dumps(outcome.payload))
    return 0


def _print_command_results(results) -> None:
    for result in results:
        status = "ok" if result.ok else "failed"
        print(f"{status}\t{result.detail}")


def _format_status(status: dict) -> str:
    lines: list[str] = [f"state\t{'ready' if status['ready'] else 'not-ready'}"]
    service = status["service"]
    lines.append(
        "service\t" + f"installed={service.get('installed')} active={service.get('active')} {service.get('detail')}"
    )
    health = status["health"]
    lines.append("health\t" + json.dumps(health, sort_keys=True))
    clipboard = status["clipboard"]
    lines.append("clipboard\t" + json.dumps(clipboard, sort_keys=True))
    paste = status["paste"]
    lines.append("paste\t" + json.dumps(paste, sort_keys=True))
    shortcut = status.get("shortcut")
    if shortcut is not None:
        lines.append("shortcut\t" + json.dumps(shortcut, sort_keys=True))
    last_event = status.get("last_log_event")
    lines.append(
        "last_log_event\t" + (json.dumps(last_event, sort_keys=True) if last_event else f"none at {toggle_log_path()}")
    )
    return "\n".join(lines)


def _format_history(events: list[dict]) -> str:
    lines = []
    for index, event in enumerate(events, start=1):
        timestamp = event.get("timestamp", "")
        source = event.get("source", "")
        text = str(event.get("text", "")).replace("\n", " ")
        lines.append(f"{index}\t{timestamp}\t{source}\t{text}")
    return "\n".join(lines)


def _copy_history(events: list[dict], index: int) -> int:
    if index < 1 or index > len(events):
        print(f"history entry {index} is not in the displayed range", file=sys.stderr)
        return 1
    text = str(events[index - 1].get("text") or "")
    try:
        SystemClipboard().write(text)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"copied\thistory {index}")
    return 0


def _format_model_rows(rows: list[ModelRow]) -> str:
    lines = ["model_id\tbackend\tmarker\tcached\tcache_path"]
    for row in rows:
        lines.append(f"{row.model_id}\t{row.backend}\t{row.marker}\t{row.cached}\t{row.cache_path}")
    return "\n".join(lines)
