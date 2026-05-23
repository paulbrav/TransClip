from __future__ import annotations

import argparse

from transclip.desktop.hotkey import install_shortcut
from transclip.settings import Settings


def handle_gnome_shortcut(args: argparse.Namespace, settings: Settings) -> int:
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
