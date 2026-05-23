from __future__ import annotations

import argparse
import sys

from transclip.product import CLI_COMMAND
from transclip.settings import Settings, get_setting, set_setting, settings_field_names


def handle_config(args: argparse.Namespace, settings: Settings) -> int:
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
