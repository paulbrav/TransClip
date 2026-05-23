from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transclip.daemon import logs_dir
from transclip.desktop.hotkey import install_shortcut
from transclip.doctor import checks_as_json, checks_as_text, run_checks
from transclip.platform.runtime import get_runtime
from transclip.settings import Settings


def handle_doctor(args: argparse.Namespace, settings: Settings, config_dir: Path | None) -> int:
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
