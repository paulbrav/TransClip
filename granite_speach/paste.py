from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Protocol

from .platform_capabilities import available_paste_backend, clipboard_capability, paste_commands, session_info
from .settings import Settings


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


class SystemClipboard:
    def __init__(self) -> None:
        self.backend = detect_clipboard_backend()

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def read(self) -> str:
        return subprocess.check_output(self.backend.read_command, text=True)

    def write(self, text: str) -> None:
        subprocess.run(self.backend.write_command, input=text, text=True, check=True)


class SystemPasteInjector:
    def __init__(self) -> None:
        self.backend_name: str | None = None

    def paste(self) -> bool:
        _LAST_PASTE_ERRORS.clear()
        self.backend_name = None
        info = session_info(environ=os.environ, system=platform.system())
        for command in paste_commands(which=shutil.which, info=info):
            if self._try(command.backend, command.command):
                return True
        return False

    def available_backend(self) -> str | None:
        info = session_info(environ=os.environ, system=platform.system())
        return available_paste_backend(which=shutil.which, info=info)

    def error_detail(self) -> str:
        return "; ".join(_LAST_PASTE_ERRORS)

    def _try(self, backend_name: str, command: list[str]) -> bool:
        if run_paste_command(command):
            self.backend_name = backend_name
            return True
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
        _LAST_PASTE_ERRORS.append(injector_exception)
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


_LAST_PASTE_ERRORS: list[str] = []


def run_paste_command(command: list[str]) -> bool:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    detail = result.stdout.strip() or f"exit status {result.returncode}"
    _LAST_PASTE_ERRORS.append(f"{command[0]} paste command failed: {detail}")
    return False


def paste_error_detail(injector: PasteInjector) -> str:
    detail = getattr(injector, "error_detail", None)
    if callable(detail):
        return str(detail())
    return ""


def detect_clipboard_backend() -> ClipboardBackend:
    info = session_info(environ=os.environ, system=platform.system())
    capability = clipboard_capability(which=shutil.which, info=info)
    if not capability.ok:
        raise RuntimeError(capability.detail)
    assert capability.backend and capability.read_command and capability.write_command
    return ClipboardBackend(capability.backend, capability.read_command, capability.write_command)
