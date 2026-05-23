from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from urllib.error import URLError

from .audio import recording_debug
from .client import InferenceClient
from .daemon import last_toggle_log_event, service_state
from .daemon_lifecycle import toggle_log_path
from .doctor_asr import (
    build_backend_checks,
    check_asr_runtime,
    check_model_cache,
    check_torch_runtime,
)
from .doctor_platform import (
    check_hotkey_readiness,
    check_microphone_devices,
    check_tcc_permissions,
)
from .doctor_types import Check
from .paste import clipboard_capability, paste_capability
from .platform_capabilities import session_info
from .platform_runtime import PlatformRuntime, get_runtime
from .settings import Settings, default_config_dir, settings_path

__all__ = [
    "Check",
    "build_backend_checks",
    "check_asr_runtime",
    "check_audio_debug",
    "check_config_files",
    "check_hotkey_readiness",
    "check_microphone_devices",
    "check_model_cache",
    "check_paste_tools",
    "check_torch_runtime",
    "checks_as_json",
    "checks_as_text",
    "run_checks",
]


def run_checks(
    settings: Settings,
    config_dir: Path | None = None,
    include_audio_debug: bool = False,
    runtime: PlatformRuntime | None = None,
) -> list[Check]:
    platform_runtime = get_runtime(runtime)
    current_service_state = service_state(runtime=platform_runtime)
    checks = [
        check_config_files(config_dir),
        check_service_manager(current_service_state, platform_runtime),
        check_service_active(current_service_state),
        check_service_health(settings),
        check_session_type(platform_runtime),
        check_clipboard_tools(platform_runtime),
        check_paste_tools(platform_runtime),
        *build_backend_checks(settings, platform_runtime),
        check_hotkey_readiness(settings, platform_runtime),
        check_microphone_devices(settings, platform_runtime),
        check_tcc_permissions(platform_runtime),
        check_last_shortcut_log_event(platform_runtime),
    ]
    if include_audio_debug:
        checks.append(check_audio_debug(settings))
    return checks


def check_config_files(config_dir: Path | None = None) -> Check:
    missing = [str(settings_path(config_dir))] if not settings_path(config_dir).exists() else []
    if not missing:
        return Check("config_files", True, f"found files in {config_dir or default_config_dir()}")
    return Check("config_files", False, "missing: " + ", ".join(missing))


def check_clipboard_tools(runtime: PlatformRuntime | None = None) -> Check:
    capability = clipboard_capability(runtime=runtime)
    return Check("clipboard_tools", capability.ok, capability.detail)


def check_paste_tools(runtime: PlatformRuntime | None = None) -> Check:
    capability = paste_capability(runtime=runtime)
    return Check("paste_tools", capability.ok, capability.detail)


def check_service_manager(
    state: dict | None = None,
    runtime: PlatformRuntime | None = None,
) -> Check:
    platform_runtime = get_runtime(runtime)
    state = state or service_state(runtime=platform_runtime)
    system = platform_runtime.system()
    if system == "Linux":
        installed = bool(state["installed"])
        return Check(
            "service_manager",
            installed,
            "systemd user unit installed" if installed else "missing systemd user unit; run: transclip install",
        )
    if system == "Darwin":
        installed = bool(state["installed"])
        return Check(
            "service_manager",
            installed,
            "LaunchAgent installed" if installed else "missing LaunchAgent; run: transclip install",
        )
    return Check("service_manager", True, f"not checked on {system}")


def check_service_active(state: dict | None = None) -> Check:
    state = state or service_state()
    return Check(
        "service_active",
        bool(state["active"]),
        f"active={state['active']}; {state['detail']}",
    )


def check_service_health(settings: Settings) -> Check:
    try:
        health = InferenceClient(settings).health()
    except URLError as exc:
        return Check("service_health", False, f"/health failed: {exc}")
    except Exception as exc:
        return Check("service_health", False, f"/health failed: {exc}")
    status = health.get("status")
    return Check(
        "service_health",
        status in {"ready", "recording"},
        f"/health status={status}; asr={health.get('asr_backend')}; cleanup={health.get('cleanup_backend')}",
    )


def check_session_type(runtime: PlatformRuntime | None = None) -> Check:
    info = session_info(runtime=runtime)
    if info.system not in {"Linux", "Darwin"}:
        return Check("session_type", True, f"not checked on {info.system}")
    if info.system == "Darwin":
        return Check("session_type", True, "macOS")
    return Check("session_type", info.session != "unknown", f"session={info.session}; desktop={info.desktop}")


def check_last_shortcut_log_event(runtime: PlatformRuntime | None = None) -> Check:
    event = last_toggle_log_event(toggle_log_path(runtime))
    if event is None:
        return Check(
            "last_shortcut_log_event",
            False,
            f"no toggle log at {toggle_log_path(runtime)}",
            required=False,
        )
    action = event.get("action") or event.get("unparsed", "unknown")
    return Check(
        "last_shortcut_log_event",
        "unparsed" not in event,
        f"last action={action}; log={toggle_log_path(runtime)}",
        required=False,
    )


def check_audio_debug(settings: Settings) -> Check:
    try:
        measurement = recording_debug(settings)
    except Exception as exc:
        return Check("audio_debug", False, f"recording debug failed: {exc}")
    detail = (
        f"device={measurement['device']}; sample_rate={measurement['sample_rate']}; "
        f"channels={measurement['channel_count']}; frames={measurement['frame_count']}; "
        f"duration={measurement['duration']:.3f}; peak={measurement['peak_amplitude']:.3f}; "
        f"rms={measurement['rms_amplitude']:.3f}; silent={measurement['silent']}"
    )
    return Check("audio_debug", not measurement["silent"], detail)


def checks_as_json(checks: list[Check]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2)


def checks_as_text(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        status = "ok" if check.ok else "missing"
        prefix = "" if check.required else "info\t"
        lines.append(f"{prefix}{status}\t{check.name}\t{check.detail}")
    return "\n".join(lines)
