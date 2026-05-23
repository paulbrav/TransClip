from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from transclip.platform.runtime import PlatformRuntime
from transclip.settings import Settings

from .common import CommandResult, ServiceState

Runner = Callable[..., subprocess.CompletedProcess[str]]


class PlatformDaemon(Protocol):
    def install(
        self,
        *,
        settings_path: Path | None,
        settings: Settings,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]: ...

    def uninstall(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> list[CommandResult]: ...

    def service_action(
        self,
        action: str,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> CommandResult: ...

    def service_state(
        self,
        *,
        runner: Runner,
        runtime: PlatformRuntime | None,
    ) -> ServiceState: ...
