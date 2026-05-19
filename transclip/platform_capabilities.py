from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .platform_runtime import PlatformRuntime, get_runtime


@dataclass(frozen=True, slots=True)
class SessionInfo:
    system: str
    session: str
    desktop: str


def session_info(
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
    runtime: PlatformRuntime | None = None,
) -> SessionInfo:
    platform_runtime = get_runtime(runtime)
    env = environ or platform_runtime.env_snapshot()
    detected_system = system or platform_runtime.system()
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
