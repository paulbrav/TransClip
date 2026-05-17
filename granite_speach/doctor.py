from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError

from .audio import recording_debug
from .client import InferenceClient
from .daemon import last_toggle_log_event, service_state, toggle_log_path
from .device import torch_cuda_usable, torch_mps_available
from .gnome_shortcut import (
    GRANITE_SHORTCUT_BINDING,
    get_gnome_shortcut_status,
)
from .models import model_cache_root, required_model_cache_paths
from .platform_capabilities import clipboard_capability, paste_capability, session_info
from .settings import Settings, default_config_dir, keywords_path, settings_path


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_checks(
    settings: Settings,
    config_dir: Path | None = None,
    include_audio_debug: bool = False,
) -> list[Check]:
    checks = [
        check_config_files(config_dir),
        check_service_manager(),
        check_service_active(),
        check_service_health(settings),
        check_session_type(),
        check_clipboard_tools(),
        check_paste_tools(),
        check_hotkey_readiness(),
        check_microphone_devices(),
        check_model_cache(settings),
        check_torch_runtime(settings),
        check_asr_runtime(settings),
        check_last_shortcut_log_event(),
    ]
    if include_audio_debug:
        checks.append(check_audio_debug(settings))
    return checks


def check_config_files(config_dir: Path | None = None) -> Check:
    missing = [str(path) for path in (settings_path(config_dir), keywords_path(config_dir)) if not path.exists()]
    if not missing:
        return Check("config_files", True, f"found files in {config_dir or default_config_dir()}")
    return Check("config_files", False, "missing: " + ", ".join(missing))


def check_clipboard_tools() -> Check:
    capability = clipboard_capability(which=shutil.which, info=session_info(system=platform.system()))
    return Check("clipboard_tools", capability.ok, capability.detail)


def check_paste_tools() -> Check:
    capability = paste_capability(
        runner=subprocess.run,
        which=shutil.which,
        info=session_info(
            environ={
                "XDG_SESSION_TYPE": os_environ("XDG_SESSION_TYPE") or "",
                "XDG_CURRENT_DESKTOP": os_environ("XDG_CURRENT_DESKTOP") or "",
                "XDG_SESSION_DESKTOP": os_environ("XDG_SESSION_DESKTOP") or "",
                "DESKTOP_SESSION": os_environ("DESKTOP_SESSION") or "",
            },
            system=platform.system(),
        ),
    )
    return Check("paste_tools", capability.ok, capability.detail)


def check_service_manager() -> Check:
    system = platform.system()
    if system == "Linux":
        installed = bool(service_state()["installed"])
        return Check(
            "service_manager",
            installed,
            "systemd user unit installed" if installed else "missing systemd user unit; run: granite-speach install",
        )
    if system == "Darwin":
        installed = bool(service_state()["installed"])
        return Check(
            "service_manager",
            installed,
            "LaunchAgent installed" if installed else "missing LaunchAgent; run: granite-speach install",
        )
    return Check("service_manager", True, f"not checked on {system}")


def check_service_active() -> Check:
    state = service_state()
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


def check_session_type() -> Check:
    info = session_info(
        environ={
            "XDG_SESSION_TYPE": os_environ("XDG_SESSION_TYPE") or "",
            "XDG_CURRENT_DESKTOP": os_environ("XDG_CURRENT_DESKTOP") or "",
            "XDG_SESSION_DESKTOP": os_environ("XDG_SESSION_DESKTOP") or "",
            "DESKTOP_SESSION": os_environ("DESKTOP_SESSION") or "",
        },
        system=platform.system(),
    )
    if info.system not in {"Linux", "Darwin"}:
        return Check("session_type", True, f"not checked on {info.system}")
    if info.system == "Darwin":
        return Check("session_type", True, "macOS")
    return Check("session_type", info.session != "unknown", f"session={info.session}; desktop={info.desktop}")


def check_last_shortcut_log_event() -> Check:
    event = last_toggle_log_event()
    if event is None:
        return Check("last_shortcut_log_event", False, f"no toggle log at {toggle_log_path()}")
    action = event.get("action") or event.get("unparsed", "unknown")
    return Check("last_shortcut_log_event", "unparsed" not in event, f"last action={action}; log={toggle_log_path()}")


def check_hotkey_readiness() -> Check:
    system = platform.system()
    if system == "Darwin":
        return Check(
            "hotkey_readiness",
            True,
            "macOS hotkey helper is not installed by the Python daemon",
        )
    if system != "Linux":
        return Check("hotkey_readiness", True, f"not checked on {system}")

    info = session_info(
        environ={
            "XDG_SESSION_TYPE": os_environ("XDG_SESSION_TYPE") or "",
            "XDG_CURRENT_DESKTOP": os_environ("XDG_CURRENT_DESKTOP") or "",
            "XDG_SESSION_DESKTOP": os_environ("XDG_SESSION_DESKTOP") or "",
            "DESKTOP_SESSION": os_environ("DESKTOP_SESSION") or "",
        },
        system=platform.system(),
    )
    if not shutil.which("gsettings"):
        return Check(
            "hotkey_readiness",
            False,
            f"session={info.session}; desktop={info.desktop}; GNOME shortcut setup requires gsettings",
        )

    try:
        status = get_gnome_shortcut_status()
    except subprocess.CalledProcessError as exc:
        return Check(
            "hotkey_readiness",
            False,
            f"session={info.session}; desktop={info.desktop}; could not inspect GNOME custom shortcuts: {exc}",
        )

    detail = (
        f"session={info.session}; desktop={info.desktop}; installed={status.installed}; "
        f"binding={status.binding or 'missing'}; command_exists={status.command_exists}"
    )
    if not status.installed:
        return Check(
            "hotkey_readiness",
            False,
            detail + "; run: granite-speach install-gnome-shortcut",
        )
    if status.binding != GRANITE_SHORTCUT_BINDING:
        return Check(
            "hotkey_readiness",
            False,
            detail + f"; expected binding={GRANITE_SHORTCUT_BINDING}",
        )
    if not status.command_exists:
        return Check(
            "hotkey_readiness",
            False,
            detail + f"; command={status.command or 'missing'}",
        )
    return Check(
        "hotkey_readiness",
        True,
        detail + f"; command={status.command}",
    )


def check_microphone_devices() -> Check:
    system = platform.system()
    if system == "Darwin":
        return Check(
            "microphone_devices",
            True,
            "macOS microphone visibility is checked by the operating system permission prompt",
        )
    if system != "Linux":
        return Check("microphone_devices", True, f"not checked on {system}")

    arecord = shutil.which("arecord")
    if arecord:
        result = subprocess.run(
            [arecord, "-l"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        if result.returncode == 0 and "card " in output and "device " in output:
            devices = [line.strip() for line in output.splitlines() if line.strip().startswith("card ")]
            return Check("microphone_devices", True, "found: " + "; ".join(devices))
        return Check(
            "microphone_devices",
            False,
            "arecord did not list capture devices" + (f": {output}" if output else ""),
        )

    if shutil.which("wpctl"):
        result = subprocess.run(
            ["wpctl", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode == 0 and "Sources:" in result.stdout:
            return Check("microphone_devices", True, "wpctl reports audio sources")
    return Check("microphone_devices", False, "requires arecord or wpctl to inspect microphone devices")


def check_model_cache(settings: Settings) -> Check:
    cache_root = model_cache_root(settings)
    required_paths = required_model_cache_paths(settings)
    missing = [str(path) for path in required_paths if not path.exists()]
    if not missing:
        return Check("model_cache", True, f"found model artifacts under {cache_root}")
    return Check(
        "model_cache",
        False,
        "missing local model artifacts: "
        + ", ".join(missing)
        + f"; run: granite-speach models prefetch --model {settings.asr_model}",
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


def check_torch_runtime(settings: Settings) -> Check:
    try:
        import torch
    except ImportError:
        return Check("torch_runtime", False, "torch is not installed; install granite-speach[models]")
    requested = settings.asr_device.lower()
    cuda_usable = torch_cuda_usable()
    mps_usable = torch_mps_available()
    version = getattr(torch, "__version__", "unknown")
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if requested in {"cuda", "rocm"} and not cuda_usable:
        return Check(
            "torch_runtime",
            False,
            f"torch {version} hip={hip}; requested {settings.asr_device}, but GPU tensor smoke failed",
        )
    if requested == "mps" and not mps_usable:
        return Check("torch_runtime", False, f"torch {version}; requested MPS, but MPS is unavailable")
    if cuda_usable:
        return Check("torch_runtime", True, f"torch {version} hip={hip}; GPU tensor smoke passed")
    if mps_usable:
        return Check("torch_runtime", True, f"torch {version}; MPS available")
    return Check("torch_runtime", True, f"torch {version} hip={hip}; auto will use CPU")


def check_asr_runtime(settings: Settings) -> Check:
    if settings.asr_backend not in {"granite_nar", "granite-nar", "nar"}:
        return Check("asr_runtime", True, f"{settings.asr_backend} has no extra runtime checks")
    try:
        import os

        import torch
    except ImportError:
        return Check("asr_runtime", False, "torch is not installed")
    if getattr(torch.version, "hip", None):
        os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        return Check("asr_runtime", False, f"Granite NAR requires flash-attn; import failed: {exc}")
    return Check("asr_runtime", True, "Granite NAR flash-attn runtime import passed")


def os_environ(name: str) -> str | None:
    import os

    return os.environ.get(name)


def checks_as_json(checks: list[Check]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2)


def checks_as_text(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        status = "ok" if check.ok else "missing"
        lines.append(f"{status}\t{check.name}\t{check.detail}")
    return "\n".join(lines)
