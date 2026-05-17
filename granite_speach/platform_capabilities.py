from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass

Runner = Callable[..., subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class SessionInfo:
    system: str
    session: str
    desktop: str


@dataclass(frozen=True, slots=True)
class ClipboardCapability:
    ok: bool
    detail: str
    backend: str | None = None
    read_command: list[str] | None = None
    write_command: list[str] | None = None


@dataclass(frozen=True, slots=True)
class PasteCommand:
    backend: str
    command: list[str]


@dataclass(frozen=True, slots=True)
class PasteCapability:
    ok: bool
    detail: str
    backend: str | None = None


def session_info(
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
) -> SessionInfo:
    env = environ or os.environ
    detected_system = system or platform.system()
    if detected_system == "Darwin":
        return SessionInfo(detected_system, "macos", "macOS")
    session = (env.get("XDG_SESSION_TYPE") or "unknown").lower()
    if session == "unknown" and env.get("WAYLAND_DISPLAY"):
        session = "wayland"
    elif session == "unknown" and env.get("DISPLAY"):
        session = "x11"
    desktop = (
        env.get("XDG_CURRENT_DESKTOP") or env.get("XDG_SESSION_DESKTOP") or env.get("DESKTOP_SESSION") or "unknown"
    )
    return SessionInfo(detected_system, session, desktop)


def clipboard_capability(
    which: Which = shutil.which,
    info: SessionInfo | None = None,
) -> ClipboardCapability:
    info = info or session_info()
    if info.system == "Darwin":
        if which("pbcopy") and which("pbpaste"):
            return ClipboardCapability(True, "found pbcopy/pbpaste", "pbcopy/pbpaste", ["pbpaste"], ["pbcopy"])
        return ClipboardCapability(False, "macOS clipboard requires pbcopy and pbpaste")
    if info.session == "wayland":
        if which("wl-copy") and which("wl-paste"):
            return ClipboardCapability(
                True,
                "found wl-clipboard",
                "wl-clipboard",
                ["wl-paste", "--no-newline"],
                ["wl-copy"],
            )
        return ClipboardCapability(
            False,
            "Wayland clipboard requires wl-clipboard. Install: sudo apt install wl-clipboard",
        )
    if which("xclip"):
        return ClipboardCapability(
            True,
            "found xclip",
            "xclip",
            ["xclip", "-selection", "clipboard", "-o"],
            ["xclip", "-selection", "clipboard"],
        )
    if which("xsel"):
        return ClipboardCapability(
            True,
            "found xsel",
            "xsel",
            ["xsel", "--clipboard", "--output"],
            ["xsel", "--clipboard", "--input"],
        )
    if which("wl-copy") and which("wl-paste"):
        return ClipboardCapability(
            True,
            "found wl-clipboard",
            "wl-clipboard",
            ["wl-paste", "--no-newline"],
            ["wl-copy"],
        )
    return ClipboardCapability(False, "No supported clipboard reader/writer found")


def paste_commands(
    which: Which = shutil.which,
    info: SessionInfo | None = None,
) -> list[PasteCommand]:
    info = info or session_info()
    if info.system == "Darwin":
        script = 'tell application "System Events" to keystroke "v" using command down'
        return [PasteCommand("osascript", ["osascript", "-e", script])] if which("osascript") else []
    if info.session == "wayland":
        commands = []
        if which("wtype"):
            commands.append(PasteCommand("wtype", ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]))
        if which("ydotool"):
            commands.append(PasteCommand("ydotool", ["ydotool", "key", "ctrl+v"]))
        return commands
    commands = []
    if which("xdotool"):
        commands.append(PasteCommand("xdotool", ["xdotool", "key", "ctrl+v"]))
    if which("ydotool"):
        commands.append(PasteCommand("ydotool", ["ydotool", "key", "ctrl+v"]))
    if which("wtype"):
        commands.append(PasteCommand("wtype", ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]))
    return commands


def available_paste_backend(
    which: Which = shutil.which,
    info: SessionInfo | None = None,
) -> str | None:
    commands = paste_commands(which=which, info=info)
    return commands[0].backend if commands else None


def paste_capability(
    runner: Runner = subprocess.run,
    which: Which = shutil.which,
    info: SessionInfo | None = None,
) -> PasteCapability:
    info = info or session_info()
    if info.system == "Darwin":
        ok = bool(which("osascript"))
        return PasteCapability(
            ok,
            "found osascript" if ok else "requires osascript and Accessibility permission",
            "osascript" if ok else None,
        )
    wtype = which("wtype")
    xdotool = which("xdotool")
    ydotool = which("ydotool")
    if info.session == "wayland":
        details = []
        if wtype:
            result = runner(
                ["wtype", "-M", "ctrl", "-m", "ctrl"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return PasteCapability(True, "wtype found and compositor accepts virtual keyboard events", "wtype")
            detail = result.stdout.strip() or f"exit status {result.returncode}"
            details.append(f"wtype unusable: {detail}")
        if ydotool:
            return PasteCapability(
                True,
                "ydotool found; requires ydotoold/uinput permissions to inject paste",
                "ydotool",
            )
        if not details:
            return PasteCapability(False, "Wayland paste injection requires wtype or ydotool; apt: wtype ydotool")
        return PasteCapability(
            False,
            "; ".join(details) + "; fallback requires ydotool with ydotoold/uinput permissions",
        )
    if xdotool:
        return PasteCapability(True, "found: xdotool", "xdotool")
    if ydotool:
        return PasteCapability(True, "found: ydotool", "ydotool")
    if wtype:
        return PasteCapability(
            False,
            "wtype is installed, but non-Wayland sessions require xdotool or ydotool; apt: xdotool ydotool",
        )
    return PasteCapability(False, "requires wtype, xdotool, or ydotool; apt: xdotool ydotool")
