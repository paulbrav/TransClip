from __future__ import annotations

import subprocess

from transclip.platform.runtime import PlatformRuntime, get_runtime


def notify(title: str, message: str, runtime: PlatformRuntime | None = None) -> bool:
    platform_runtime = get_runtime(runtime)
    try:
        if platform_runtime.system() == "Darwin":
            script = f'display notification "{_escape(message)}" with title "{_escape(title)}"'
            platform_runtime.run(["osascript", "-e", script], check=True)
            return True
        if platform_runtime.which("notify-send"):
            platform_runtime.run(["notify-send", title, message], check=True)
            return True
    except subprocess.CalledProcessError:
        return False
    return False


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
