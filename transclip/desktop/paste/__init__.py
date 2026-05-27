from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from transclip.platform.capabilities import SessionInfo, session_info
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.settings import Settings

from .platform import (
    resolve_clipboard_capability,
    resolve_paste_capability,
    resolve_paste_commands,
)

Which = Callable[[str], str | None]

PASTE_COMMAND_TIMEOUT_SECONDS = 5.0


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
    read_fn: Callable[[], str] | None = None
    write_fn: Callable[[str], None] | None = None


@dataclass(frozen=True, slots=True)
class PasteCommand:
    backend: str
    command: list[str]
    native: Callable[[], None] | None = None


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
        return any(self._try(command) for command in paste_commands(runtime=self.runtime))

    def available_backend(self) -> str | None:
        return available_paste_backend(runtime=self.runtime)

    def error_detail(self) -> str:
        return "; ".join(self.errors)

    def _try(self, paste_command: PasteCommand) -> bool:
        if paste_command.native is not None:
            try:
                paste_command.native()
            except Exception as exc:
                self.errors.append(f"{paste_command.backend} paste failed: {exc}")
                return False
            self.backend_name = paste_command.backend
            return True
        error = run_paste_command(paste_command.command, runtime=self.runtime)
        if error is None:
            self.backend_name = paste_command.backend
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
    return resolve_clipboard_capability(which, info)


def paste_commands(
    which: Which | None = None,
    info: SessionInfo | None = None,
    runtime: PlatformRuntime | None = None,
) -> list[PasteCommand]:
    platform_runtime = get_runtime(runtime)
    which = which or platform_runtime.which
    info = info or session_info(runtime=platform_runtime)
    return resolve_paste_commands(which, info)


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
    return resolve_paste_capability(which, info, runner)


def run_paste_command(command: list[str], runtime: PlatformRuntime | None = None) -> str | None:
    try:
        result = get_runtime(runtime).run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=PASTE_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            f"{command[0]} paste command timed out after "
            f"{PASTE_COMMAND_TIMEOUT_SECONDS:g}s"
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
    return ClipboardBackend(
        capability.backend,
        read_command,
        write_command,
        read_fn=capability.read_fn,
        write_fn=capability.write_fn,
    )
