from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Literal

from transclip.platform.capabilities import SessionInfo, session_info
from transclip.platform.runtime import PlatformRuntime, get_runtime

FocusKind = Literal["terminal", "gui", "unknown"]

DEFAULT_TERMINAL_WM_CLASS_PATTERNS = (
    r"gnome-terminal",
    r"org\.gnome\.terminal",
    r"kitty",
    r"alacritty",
    r"wezterm",
    r"org\.wezfurlong\.wezterm",
    r"ghostty",
    r"foot",
    r"tilix",
    r"konsole",
    r"cursor",
    r"code-oss",
    r"code",
)


@dataclass(frozen=True, slots=True)
class FocusedApp:
    kind: FocusKind
    wm_class: str = ""


def parse_terminal_wm_class_patterns(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return DEFAULT_TERMINAL_WM_CLASS_PATTERNS
    patterns: list[str] = []
    for part in raw.split(","):
        pattern = part.strip().lower()
        if pattern:
            patterns.append(pattern)
    return tuple(patterns or DEFAULT_TERMINAL_WM_CLASS_PATTERNS)


def classify_wm_class(wm_class: str, patterns: tuple[str, ...]) -> FocusKind:
    normalized = wm_class.strip().lower()
    if not normalized:
        return "unknown"
    for pattern in patterns:
        if re.search(pattern, normalized):
            return "terminal"
    return "gui"


def detect_focused_app(
    runtime: PlatformRuntime | None = None,
    *,
    terminal_patterns: tuple[str, ...] | None = None,
) -> FocusedApp:
    platform_runtime = get_runtime(runtime)
    info = session_info(runtime=platform_runtime)
    if info.system != "Linux":
        return FocusedApp(kind="unknown")
    patterns = terminal_patterns or DEFAULT_TERMINAL_WM_CLASS_PATTERNS
    wm_class = _read_focused_wm_class(platform_runtime, info)
    return FocusedApp(kind=classify_wm_class(wm_class, patterns), wm_class=wm_class)


def _read_focused_wm_class(runtime: PlatformRuntime, info: SessionInfo) -> str:
    if "gnome" not in info.desktop.lower():
        return ""
    script = "global.display.focus_window?.get_wm_class()?.toLowerCase() ?? ''"
    try:
        result = runtime.run(
            [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.gnome.Shell",
                "--object-path",
                "/org/gnome/Shell",
                "--method",
                "org.gnome.Shell.Eval",
                script,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return _parse_gnome_eval_output(result.stdout)


def _parse_gnome_eval_output(output: str) -> str:
    match = re.search(r"\(\s*(true|false)\s*,\s*'([^']*)'\s*\)", output, re.IGNORECASE)
    if match and match.group(1).lower() == "true":
        return match.group(2).strip().lower()
    match = re.search(r'\(\s*(true|false)\s*,\s*"([^"]*)"\s*\)', output, re.IGNORECASE)
    if match and match.group(1).lower() == "true":
        return match.group(2).strip().lower()
    return ""
