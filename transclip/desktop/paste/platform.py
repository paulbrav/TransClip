from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from transclip.platform.capabilities import SessionInfo

if TYPE_CHECKING:
    from transclip.desktop.paste import ClipboardCapability, PasteCapability, PasteCommand

Which = Callable[[str], str | None]
Runner = Callable[..., subprocess.CompletedProcess[str]]

TERMINAL_PASTE_SHORTCUT = "ctrl+shift+v"
WTYPE_TERMINAL_PASTE_COMMAND = ("wtype", "-M", "ctrl", "-M", "shift", "v", "-m", "shift", "-m", "ctrl")
SENDINPUT_PASTE_BACKEND = "sendinput"


def _win32_read_clipboard() -> str:
    from .win32 import read_clipboard_text

    return read_clipboard_text()


def _win32_write_clipboard(text: str) -> None:
    from .win32 import write_clipboard_text

    write_clipboard_text(text)


def _win32_sendinput_paste() -> None:
    from .win32 import send_ctrl_v_paste

    send_ctrl_v_paste()


@dataclass(frozen=True, slots=True)
class ClipboardSpec:
    backend: str
    detail_ok: str
    read_command: list[str]
    write_command: list[str]
    available: Callable[[Which, SessionInfo], bool]
    detail_fail: str = ""
    read_fn: Callable[[], str] | None = None
    write_fn: Callable[[str], None] | None = None


@dataclass(frozen=True, slots=True)
class PasteSpec:
    backend: str
    capability: Callable[[Which, SessionInfo, Runner], PasteCapability | None]
    build_commands: Callable[[Which, SessionInfo], list[list[str]]] | None = None
    execute_native: Callable[[], None] | None = None
    failure_detail: Callable[[list[str]], str] | None = None


def _darwin_clipboard_specs() -> Sequence[ClipboardSpec]:
    return (
        ClipboardSpec(
            "pbcopy/pbpaste",
            "found pbcopy/pbpaste",
            ["pbpaste"],
            ["pbcopy"],
            lambda which, _info: bool(which("pbcopy") and which("pbpaste")),
            "macOS clipboard requires pbcopy and pbpaste",
        ),
    )


def _windows_clipboard_specs() -> Sequence[ClipboardSpec]:
    return (
        ClipboardSpec(
            "win32",
            "native Win32 clipboard available",
            [],
            [],
            lambda _which, info: info.system == "Windows",
            read_fn=_win32_read_clipboard,
            write_fn=_win32_write_clipboard,
        ),
    )


def _linux_clipboard_specs() -> Sequence[ClipboardSpec]:
    return (
        ClipboardSpec(
            "wl-clipboard",
            "found wl-clipboard",
            ["wl-paste", "--no-newline"],
            ["wl-copy"],
            lambda which, info: info.session == "wayland" and bool(which("wl-copy") and which("wl-paste")),
            "Wayland clipboard requires wl-clipboard. Install: sudo apt install wl-clipboard",
        ),
        ClipboardSpec(
            "xclip",
            "found xclip",
            ["xclip", "-selection", "clipboard", "-o"],
            ["xclip", "-selection", "clipboard"],
            lambda which, info: info.session != "wayland" and bool(which("xclip")),
        ),
        ClipboardSpec(
            "xsel",
            "found xsel",
            ["xsel", "--clipboard", "--output"],
            ["xsel", "--clipboard", "--input"],
            lambda which, info: info.session != "wayland" and bool(which("xsel")),
        ),
        ClipboardSpec(
            "wl-clipboard",
            "found wl-clipboard",
            ["wl-paste", "--no-newline"],
            ["wl-copy"],
            lambda which, info: info.session != "wayland" and bool(which("wl-copy") and which("wl-paste")),
        ),
    )


def _darwin_paste_specs() -> Sequence[PasteSpec]:
    script = 'tell application "System Events" to keystroke "v" using command down'

    def build_commands(which: Which, _info: SessionInfo) -> list[list[str]]:
        return [["osascript", "-e", script]] if which("osascript") else []

    def capability(which: Which, _info: SessionInfo, _runner: Runner):
        from transclip.desktop.paste import PasteCapability

        ok = bool(which("osascript"))
        return PasteCapability(
            ok,
            "found osascript" if ok else "requires osascript and Accessibility permission",
            "osascript" if ok else None,
        )

    def darwin_failure(_details: list[str]) -> str:
        return "requires osascript and Accessibility permission"

    return (
        PasteSpec("osascript", capability, build_commands=build_commands, failure_detail=darwin_failure),
    )


def _windows_paste_specs() -> Sequence[PasteSpec]:
    def capability(_which: Which, _info: SessionInfo, _runner: Runner):
        from transclip.desktop.paste import PasteCapability

        return PasteCapability(True, "Win32 SendInput Ctrl+V paste available", SENDINPUT_PASTE_BACKEND)

    def failure(_details: list[str]) -> str:
        return "Win32 SendInput paste unavailable"

    return (
        PasteSpec(
            SENDINPUT_PASTE_BACKEND,
            capability,
            execute_native=_win32_sendinput_paste,
            failure_detail=failure,
        ),
    )


def _wtype_paste_commands(which: Which, _info: SessionInfo) -> list[list[str]]:
    return [list(WTYPE_TERMINAL_PASTE_COMMAND)] if which("wtype") else []


def _ydotool_paste_commands(which: Which, _info: SessionInfo) -> list[list[str]]:
    return [["ydotool", "key", TERMINAL_PASTE_SHORTCUT]] if which("ydotool") else []


def _ydotool_found_capability(which: Which, _info: SessionInfo, _runner: Runner):
    from transclip.desktop.paste import PasteCapability

    if which("ydotool"):
        return PasteCapability(True, "found: ydotool", "ydotool")
    return None


def _wayland_paste_specs() -> Sequence[PasteSpec]:
    def wtype_capability(which: Which, _info: SessionInfo, runner: Runner):
        from transclip.desktop.paste import PasteCapability

        if not which("wtype"):
            return None
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
        return PasteCapability(False, f"wtype unusable: {detail}")

    def ydotool_capability(which: Which, _info: SessionInfo, _runner: Runner):
        from transclip.desktop.paste import PasteCapability

        if which("ydotool"):
            return PasteCapability(
                True,
                "ydotool found; requires ydotoold/uinput permissions to inject paste",
                "ydotool",
            )
        return None

    def wayland_failure(details: list[str]) -> str:
        if not details:
            return "Wayland paste injection requires wtype or ydotool; apt: wtype ydotool"
        return "; ".join(details) + "; fallback requires ydotool with ydotoold/uinput permissions"

    return (
        PasteSpec("wtype", wtype_capability, build_commands=_wtype_paste_commands),
        PasteSpec(
            "ydotool",
            ydotool_capability,
            build_commands=_ydotool_paste_commands,
            failure_detail=wayland_failure,
        ),
    )


def _x11_paste_specs() -> Sequence[PasteSpec]:
    def xdotool_commands(which: Which, _info: SessionInfo) -> list[list[str]]:
        return [["xdotool", "key", TERMINAL_PASTE_SHORTCUT]] if which("xdotool") else []

    def xdotool_capability(which: Which, _info: SessionInfo, _runner: Runner):
        from transclip.desktop.paste import PasteCapability

        if which("xdotool"):
            return PasteCapability(True, "found: xdotool", "xdotool")
        return None

    def wtype_capability(which: Which, _info: SessionInfo, _runner: Runner):
        from transclip.desktop.paste import PasteCapability

        if which("wtype"):
            return PasteCapability(
                False,
                "wtype is installed, but non-Wayland sessions require xdotool or ydotool; apt: xdotool ydotool",
            )
        return None

    def x11_failure(details: list[str]) -> str:
        if details:
            return details[-1]
        return "requires wtype, xdotool, or ydotool; apt: xdotool ydotool"

    return (
        PasteSpec("xdotool", xdotool_capability, build_commands=xdotool_commands),
        PasteSpec("ydotool", _ydotool_found_capability, build_commands=_ydotool_paste_commands),
        PasteSpec("wtype", wtype_capability, build_commands=_wtype_paste_commands, failure_detail=x11_failure),
    )


def clipboard_specs(info: SessionInfo) -> Sequence[ClipboardSpec]:
    if info.system == "Darwin":
        return _darwin_clipboard_specs()
    if info.system == "Windows":
        return _windows_clipboard_specs()
    return _linux_clipboard_specs()


def paste_specs(info: SessionInfo) -> Sequence[PasteSpec]:
    if info.system == "Darwin":
        return _darwin_paste_specs()
    if info.system == "Windows":
        return _windows_paste_specs()
    if info.session == "wayland":
        return _wayland_paste_specs()
    return _x11_paste_specs()


def resolve_clipboard_capability(
    which: Which,
    info: SessionInfo,
) -> ClipboardCapability:
    from transclip.desktop.paste import ClipboardCapability

    specs = clipboard_specs(info)
    failures: list[str] = []
    for spec in specs:
        if not spec.available(which, info):
            if spec.detail_fail:
                failures.append(spec.detail_fail)
            continue
        return ClipboardCapability(
            True,
            spec.detail_ok,
            spec.backend,
            list(spec.read_command),
            list(spec.write_command),
            read_fn=spec.read_fn,
            write_fn=spec.write_fn,
        )
    if failures:
        return ClipboardCapability(False, failures[0])
    return ClipboardCapability(False, "No supported clipboard reader/writer found")


def resolve_paste_commands(which: Which, info: SessionInfo) -> list[PasteCommand]:
    from transclip.desktop.paste import PasteCommand

    commands: list[PasteCommand] = []
    for spec in paste_specs(info):
        if spec.execute_native is not None:
            commands.append(PasteCommand(spec.backend, [], native=spec.execute_native))
            continue
        if spec.build_commands is None:
            continue
        for command in spec.build_commands(which, info):
            commands.append(PasteCommand(spec.backend, command))
    return commands


def resolve_paste_capability(
    which: Which,
    info: SessionInfo,
    runner: Runner,
) -> PasteCapability:
    from transclip.desktop.paste import PasteCapability

    specs = paste_specs(info)
    details: list[str] = []
    for spec in specs:
        result = spec.capability(which, info, runner)
        if result is None:
            continue
        if result.ok:
            return result
        details.append(result.detail)
    for spec in specs:
        if spec.failure_detail is not None:
            return PasteCapability(False, spec.failure_detail(details))
    return PasteCapability(False, "paste injection unavailable")
