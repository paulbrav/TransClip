from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .platform_capabilities import SessionInfo, session_info
from .platform_runtime import PlatformRuntime, get_runtime
from .settings import Settings

SENDINPUT_PASTE_BACKEND = "sendinput"

Which = Callable[[str], str | None]
TERMINAL_PASTE_SHORTCUT = "ctrl+shift+v"
WTYPE_TERMINAL_PASTE_COMMAND = ("wtype", "-M", "ctrl", "-M", "shift", "v", "-m", "shift", "-m", "ctrl")


def _win32_read_clipboard() -> str:
    from .win32_clipboard import read_clipboard_text

    return read_clipboard_text()


def _win32_write_clipboard(text: str) -> None:
    from .win32_clipboard import write_clipboard_text

    write_clipboard_text(text)


def _win32_sendinput_paste() -> None:
    from .win32_clipboard import send_ctrl_v_paste

    send_ctrl_v_paste()


PASTE_INVOKERS: dict[str, Callable[[], None]] = {
    SENDINPUT_PASTE_BACKEND: _win32_sendinput_paste,
}


class Clipboard(Protocol):
    def read(self) -> str: ...

    def write(self, text: str) -> None: ...


class PasteInjector(Protocol):
    def paste(self) -> bool: ...


@dataclass(slots=True)
class ClipboardBackend:
    name: str
    read_command: list[str]
    write_command: list[str]
    read_fn: Callable[[], str] | None = None
    write_fn: Callable[[str], None] | None = None


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


class SystemClipboard:
    def __init__(self, runtime: PlatformRuntime | None = None) -> None:
        self.runtime = get_runtime(runtime)
        self.backend = detect_clipboard_backend(self.runtime)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def read(self) -> str:
        if self.backend.read_fn is not None:
            return self.backend.read_fn()
        return self.runtime.check_output(self.backend.read_command, text=True)

    def write(self, text: str) -> None:
        if self.backend.write_fn is not None:
            self.backend.write_fn(text)
            return
        self.runtime.run(self.backend.write_command, input=text, text=True, check=True)


class SystemPasteInjector:
    def __init__(self, runtime: PlatformRuntime | None = None) -> None:
        self.runtime = get_runtime(runtime)
        self.backend_name: str | None = None
        self.errors: list[str] = []

    def paste(self) -> bool:
        self.errors.clear()
        self.backend_name = None
        return any(self._try(command.backend, command.command) for command in paste_commands(runtime=self.runtime))

    def available_backend(self) -> str | None:
        return available_paste_backend(runtime=self.runtime)

    def error_detail(self) -> str:
        return "; ".join(self.errors)

    def _try(self, backend_name: str, command: list[str]) -> bool:
        invoker = PASTE_INVOKERS.get(backend_name)
        if invoker is not None:
            try:
                invoker()
            except Exception as exc:
                self.errors.append(f"{backend_name} paste failed: {exc}")
                return False
            self.backend_name = backend_name
            return True
        error = run_paste_command(command, runtime=self.runtime)
        if error is None:
            self.backend_name = backend_name
            return True
        self.errors.append(error)
        return False


@dataclass(slots=True)
class PasteResult:
    copied: bool
    pasted: bool
    restored: bool
    transcript_left_on_clipboard: bool
    clipboard_backend: str
    paste_backend: str | None
    error_detail: str = ""


def paste_transcript(
    transcript: str,
    settings: Settings,
    clipboard: Clipboard | None = None,
    injector: PasteInjector | None = None,
) -> PasteResult:
    try:
        clipboard = clipboard or SystemClipboard()
    except Exception as exc:
        return PasteResult(
            copied=False,
            pasted=False,
            restored=False,
            transcript_left_on_clipboard=False,
            clipboard_backend="unavailable",
            paste_backend=None,
            error_detail=str(exc),
        )
    injector = injector or SystemPasteInjector()
    clipboard_backend = getattr(clipboard, "backend_name", clipboard.__class__.__name__)
    prior = ""
    prior_read_ok = False
    try:
        prior = clipboard.read()
        prior_read_ok = True
    except Exception:
        prior_read_ok = False
    try:
        clipboard.write(transcript)
    except Exception as exc:
        return PasteResult(
            copied=False,
            pasted=False,
            restored=False,
            transcript_left_on_clipboard=False,
            clipboard_backend=str(clipboard_backend),
            paste_backend=None,
            error_detail=str(exc),
        )
    copied = True
    injector_exception = ""
    try:
        pasted = injector.paste()
    except Exception as exc:
        pasted = False
        injector_exception = str(exc)
    error_detail = "" if pasted else paste_error_detail(injector) or injector_exception
    paste_backend = getattr(injector, "backend_name", None)
    restored = False
    if pasted and settings.restore_clipboard_after_paste and prior_read_ok:
        time.sleep(settings.clipboard_restore_delay_ms / 1000)
        try:
            if clipboard.read() == transcript:
                clipboard.write(prior)
                restored = True
        except RuntimeError:
            restored = False
    return PasteResult(
        copied=copied,
        pasted=pasted,
        restored=restored,
        transcript_left_on_clipboard=not restored,
        clipboard_backend=str(clipboard_backend),
        paste_backend=paste_backend,
        error_detail=error_detail,
    )


def clipboard_capability(
    which: Which | None = None,
    info: SessionInfo | None = None,
    runtime: PlatformRuntime | None = None,
) -> ClipboardCapability:
    platform_runtime = get_runtime(runtime)
    which = which or platform_runtime.which
    info = info or session_info(runtime=platform_runtime)
    if info.system == "Darwin":
        if which("pbcopy") and which("pbpaste"):
            return ClipboardCapability(True, "found pbcopy/pbpaste", "pbcopy/pbpaste", ["pbpaste"], ["pbcopy"])
        return ClipboardCapability(False, "macOS clipboard requires pbcopy and pbpaste")
    if info.system == "Windows":
        return ClipboardCapability(True, "native Win32 clipboard available", "win32", [], [])
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
    which: Which | None = None,
    info: SessionInfo | None = None,
    runtime: PlatformRuntime | None = None,
) -> list[PasteCommand]:
    platform_runtime = get_runtime(runtime)
    which = which or platform_runtime.which
    info = info or session_info(runtime=platform_runtime)
    if info.system == "Darwin":
        script = 'tell application "System Events" to keystroke "v" using command down'
        return [PasteCommand("osascript", ["osascript", "-e", script])] if which("osascript") else []
    if info.system == "Windows":
        return [PasteCommand(SENDINPUT_PASTE_BACKEND, [])]
    if info.session == "wayland":
        commands = []
        if which("wtype"):
            commands.append(PasteCommand("wtype", list(WTYPE_TERMINAL_PASTE_COMMAND)))
        if which("ydotool"):
            commands.append(PasteCommand("ydotool", ["ydotool", "key", TERMINAL_PASTE_SHORTCUT]))
        return commands
    commands = []
    if which("xdotool"):
        commands.append(PasteCommand("xdotool", ["xdotool", "key", TERMINAL_PASTE_SHORTCUT]))
    if which("ydotool"):
        commands.append(PasteCommand("ydotool", ["ydotool", "key", TERMINAL_PASTE_SHORTCUT]))
    if which("wtype"):
        commands.append(PasteCommand("wtype", list(WTYPE_TERMINAL_PASTE_COMMAND)))
    return commands


def available_paste_backend(
    which: Which | None = None,
    info: SessionInfo | None = None,
    runtime: PlatformRuntime | None = None,
) -> str | None:
    commands = paste_commands(which=which, info=info, runtime=runtime)
    return commands[0].backend if commands else None


def paste_capability(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    which: Which | None = None,
    info: SessionInfo | None = None,
    runtime: PlatformRuntime | None = None,
) -> PasteCapability:
    platform_runtime = get_runtime(runtime)
    runner = runner or platform_runtime.run
    which = which or platform_runtime.which
    info = info or session_info(runtime=platform_runtime)
    if info.system == "Darwin":
        ok = bool(which("osascript"))
        return PasteCapability(
            ok,
            "found osascript" if ok else "requires osascript and Accessibility permission",
            "osascript" if ok else None,
        )
    if info.system == "Windows":
        return PasteCapability(True, "Win32 SendInput Ctrl+V paste available", SENDINPUT_PASTE_BACKEND)
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


def run_paste_command(command: list[str], runtime: PlatformRuntime | None = None) -> str | None:
    result = get_runtime(runtime).run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    detail = result.stdout.strip() or f"exit status {result.returncode}"
    return f"{command[0]} paste command failed: {detail}"


def paste_error_detail(injector: PasteInjector) -> str:
    detail = getattr(injector, "error_detail", None)
    if callable(detail):
        return str(detail())
    return ""


def detect_clipboard_backend(runtime: PlatformRuntime | None = None) -> ClipboardBackend:
    capability = clipboard_capability(runtime=runtime)
    if not capability.ok:
        raise RuntimeError(capability.detail)
    assert capability.backend
    read_command = capability.read_command or []
    write_command = capability.write_command or []
    read_fn = _win32_read_clipboard if capability.backend == "win32" else None
    write_fn = _win32_write_clipboard if capability.backend == "win32" else None
    return ClipboardBackend(capability.backend, read_command, write_command, read_fn=read_fn, write_fn=write_fn)
