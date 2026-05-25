from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from transclip.platform.capabilities import SessionInfo, session_info
from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.settings import Settings, TextDeliveryMode

from .focus import detect_focused_app, parse_terminal_wm_class_patterns
from .platform import (
    GUI_PASTE_SHORTCUT,
    TERMINAL_PASTE_SHORTCUT,
    paste_shortcut_label,
    paste_tools_detail,
    resolve_clipboard_capability,
    resolve_paste_capability,
    resolve_paste_commands,
    resolve_paste_shortcut,
)

Which = Callable[[str], str | None]


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
    def __init__(
        self,
        runtime: PlatformRuntime | None = None,
        *,
        shortcut: str | None = None,
    ) -> None:
        self.runtime = get_runtime(runtime)
        self.shortcut = shortcut
        self.backend_name: str | None = None
        self.errors: list[str] = []

    def paste(self) -> bool:
        self.errors.clear()
        self.backend_name = None
        if not self.shortcut:
            self.errors.append("paste shortcut not configured")
            return False
        commands = self._paste_commands()
        return any(self._try(command) for command in commands)

    def _paste_commands(self) -> list[PasteCommand]:
        return resolve_paste_commands(
            self.runtime.which,
            session_info(runtime=self.runtime),
            shortcut=self.shortcut,
        )

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


@dataclass(frozen=True, slots=True)
class PasteDeliveryPlan:
    shortcut: str
    label: str
    focused_app_kind: str | None = None


@dataclass(slots=True)
class PasteResult:
    copied: bool
    pasted: bool
    injected: bool
    restored: bool
    transcript_left_on_clipboard: bool
    clipboard_backend: str
    paste_backend: str | None
    error_detail: str = ""
    paste_shortcut: str = ""
    delivery: TextDeliveryMode = "inject"
    focused_app_kind: str | None = None


def _resolve_paste_runtime(
    clipboard: Clipboard | None,
    injector: PasteInjector | None,
) -> PlatformRuntime | None:
    runtime = getattr(injector, "runtime", None)
    if runtime is not None:
        return runtime
    return getattr(clipboard, "runtime", None)


def plan_paste_delivery(
    settings: Settings,
    runtime: PlatformRuntime | None = None,
    *,
    probe_focus: bool = True,
) -> PasteDeliveryPlan:
    platform_runtime = get_runtime(runtime)
    info = session_info(runtime=platform_runtime)
    focused_kind: str | None = None
    if info.system == "Linux" and settings.focus_aware_paste and probe_focus:
        focused = detect_focused_app(
            platform_runtime,
            terminal_patterns=parse_terminal_wm_class_patterns(settings.terminal_wm_class_patterns),
        )
        focused_kind = focused.kind
        shortcut = resolve_paste_shortcut(info, focused.kind)
    elif info.system == "Linux":
        shortcut = TERMINAL_PASTE_SHORTCUT
    elif info.system == "Windows":
        shortcut = GUI_PASTE_SHORTCUT
    else:
        shortcut = "command+v"
    return PasteDeliveryPlan(
        shortcut=shortcut,
        label=paste_shortcut_label(shortcut),
        focused_app_kind=focused_kind,
    )


def paste_transcript(
    transcript: str,
    settings: Settings,
    clipboard: Clipboard | None = None,
    injector: PasteInjector | None = None,
) -> PasteResult:
    platform_runtime = get_runtime(_resolve_paste_runtime(clipboard, injector))
    try:
        clipboard = clipboard or SystemClipboard(runtime=platform_runtime)
    except Exception as exc:
        return PasteResult(
            copied=False,
            pasted=False,
            injected=False,
            restored=False,
            transcript_left_on_clipboard=False,
            clipboard_backend="unavailable",
            paste_backend=None,
            error_detail=str(exc),
        )
    plan = plan_paste_delivery(
        settings,
        runtime=platform_runtime,
        probe_focus=settings.text_delivery_mode != "clipboard_only",
    )
    if injector is None:
        injector = SystemPasteInjector(runtime=platform_runtime, shortcut=plan.shortcut)
    elif isinstance(injector, SystemPasteInjector):
        injector.shortcut = plan.shortcut
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
            injected=False,
            restored=False,
            transcript_left_on_clipboard=False,
            clipboard_backend=str(clipboard_backend),
            paste_backend=None,
            error_detail=str(exc),
        )
    copied = True
    if settings.text_delivery_mode == "clipboard_only":
        return PasteResult(
            copied=copied,
            pasted=True,
            injected=False,
            restored=False,
            transcript_left_on_clipboard=True,
            clipboard_backend=str(clipboard_backend),
            paste_backend=None,
            paste_shortcut=plan.label,
            delivery="clipboard_only",
            focused_app_kind=plan.focused_app_kind,
        )
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
        injected=pasted,
        restored=restored,
        transcript_left_on_clipboard=not restored,
        clipboard_backend=str(clipboard_backend),
        paste_backend=paste_backend,
        error_detail=error_detail,
        paste_shortcut=plan.label,
        delivery="inject",
        focused_app_kind=plan.focused_app_kind,
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
    *,
    shortcut: str | None = None,
) -> list[PasteCommand]:
    platform_runtime = get_runtime(runtime)
    which = which or platform_runtime.which
    info = info or session_info(runtime=platform_runtime)
    return resolve_paste_commands(which, info, shortcut=shortcut)


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
    capability = resolve_paste_capability(which, info, runner)
    if capability.ok:
        return PasteCapability(
            capability.ok,
            paste_tools_detail(info, capability),
            capability.backend,
        )
    return capability


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
    return ClipboardBackend(
        capability.backend,
        read_command,
        write_command,
        read_fn=capability.read_fn,
        write_fn=capability.write_fn,
    )
