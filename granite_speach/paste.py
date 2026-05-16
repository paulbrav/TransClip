from __future__ import annotations

from dataclasses import dataclass
import platform
import shutil
import subprocess
import time
from typing import Protocol

from .settings import Settings


class Clipboard(Protocol):
    def read(self) -> str:
        ...

    def write(self, text: str) -> None:
        ...


class PasteInjector(Protocol):
    def paste(self) -> bool:
        ...


class SystemClipboard:
    def read(self) -> str:
        system = platform.system()
        if system == "Darwin":
            return subprocess.check_output(["pbpaste"], text=True)
        if shutil.which("wl-paste"):
            return subprocess.check_output(["wl-paste", "--no-newline"], text=True)
        if shutil.which("xclip"):
            return subprocess.check_output(["xclip", "-selection", "clipboard", "-o"], text=True)
        if shutil.which("xsel"):
            return subprocess.check_output(["xsel", "--clipboard", "--output"], text=True)
        raise RuntimeError("No supported clipboard reader found")

    def write(self, text: str) -> None:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text, text=True, check=True)
            return
        if shutil.which("wl-copy"):
            subprocess.run(["wl-copy"], input=text, text=True, check=True)
            return
        if shutil.which("xclip"):
            subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=True)
            return
        if shutil.which("xsel"):
            subprocess.run(["xsel", "--clipboard", "--input"], input=text, text=True, check=True)
            return
        raise RuntimeError("No supported clipboard writer found")


class SystemPasteInjector:
    def paste(self) -> bool:
        _LAST_PASTE_ERRORS.clear()
        system = platform.system()
        if system == "Darwin":
            script = 'tell application "System Events" to keystroke "v" using command down'
            return run_paste_command(["osascript", "-e", script])
        if shutil.which("wtype") and run_paste_command(["wtype", "-M", "ctrl", "v", "-m", "ctrl"]):
            return True
        if shutil.which("xdotool") and run_paste_command(["xdotool", "key", "ctrl+v"]):
            return True
        if shutil.which("ydotool") and run_paste_command(["ydotool", "key", "ctrl+v"]):
            return True
        return False

    def error_detail(self) -> str:
        return "; ".join(_LAST_PASTE_ERRORS)


@dataclass(slots=True)
class PasteResult:
    pasted: bool
    restored: bool
    transcript_left_on_clipboard: bool
    error_detail: str = ""


def paste_transcript(
    transcript: str,
    settings: Settings,
    clipboard: Clipboard | None = None,
    injector: PasteInjector | None = None,
) -> PasteResult:
    clipboard = clipboard or SystemClipboard()
    injector = injector or SystemPasteInjector()
    prior = clipboard.read()
    clipboard.write(transcript)
    pasted = injector.paste()
    error_detail = "" if pasted else paste_error_detail(injector)
    restored = False
    if pasted and settings.restore_clipboard_after_paste:
        time.sleep(settings.clipboard_restore_delay_ms / 1000)
        try:
            if clipboard.read() == transcript:
                clipboard.write(prior)
                restored = True
        except RuntimeError:
            restored = False
    return PasteResult(
        pasted=pasted,
        restored=restored,
        transcript_left_on_clipboard=not restored,
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
