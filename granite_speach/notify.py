from __future__ import annotations

import platform
import shutil
import subprocess


def notify(title: str, message: str) -> bool:
    try:
        if platform.system() == "Darwin":
            script = f'display notification "{_escape(message)}" with title "{_escape(title)}"'
            subprocess.run(["osascript", "-e", script], check=True)
            return True
        if shutil.which("notify-send"):
            subprocess.run(["notify-send", title, message], check=True)
            return True
    except subprocess.CalledProcessError:
        return False
    return False


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
