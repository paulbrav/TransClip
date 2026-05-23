from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol, cast

SESSION_ENV_KEYS = (
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_DESKTOP",
    "DESKTOP_SESSION",
    "WAYLAND_DISPLAY",
    "DISPLAY",
)


class PlatformRuntime(Protocol):
    def system(self) -> str: ...

    def home_dir(self) -> Path: ...

    def environ(self, name: str, default: str | None = None) -> str | None: ...

    def env_snapshot(self, names: tuple[str, ...] = SESSION_ENV_KEYS) -> dict[str, str]: ...

    def which(self, program: str) -> str | None: ...

    def run(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]: ...

    def check_output(self, command: list[str], **kwargs: Any) -> str: ...


class DefaultPlatformRuntime:
    def system(self) -> str:
        return platform.system()

    def home_dir(self) -> Path:
        return Path.home()

    def environ(self, name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, default)

    def env_snapshot(self, names: tuple[str, ...] = SESSION_ENV_KEYS) -> dict[str, str]:
        return {name: os.environ.get(name, "") for name in names}

    def which(self, program: str) -> str | None:
        return shutil.which(program)

    def run(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return cast(subprocess.CompletedProcess[str], subprocess.run(command, **kwargs))

    def check_output(self, command: list[str], **kwargs: Any) -> str:
        kwargs.setdefault("text", True)
        output = subprocess.check_output(command, **kwargs)
        return output.decode() if isinstance(output, bytes) else output


default_platform_runtime = DefaultPlatformRuntime()


def get_runtime(runtime: PlatformRuntime | None = None) -> PlatformRuntime:
    return runtime or default_platform_runtime


def user_config_dir(app_name: str, runtime: PlatformRuntime | None = None) -> Path:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() == "Darwin":
        return platform_runtime.home_dir() / "Library" / "Application Support" / app_name
    return platform_runtime.home_dir() / ".config" / app_name


def user_cache_dir(app_name: str, runtime: PlatformRuntime | None = None) -> Path:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() == "Darwin":
        return platform_runtime.home_dir() / "Library" / "Caches" / app_name
    return platform_runtime.home_dir() / ".cache" / app_name


def user_log_dir(app_name: str, runtime: PlatformRuntime | None = None) -> Path:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() == "Darwin":
        return platform_runtime.home_dir() / "Library" / "Logs" / app_name
    return user_cache_dir(app_name, platform_runtime)
