from __future__ import annotations

import json
import platform as py_platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError

from .audio import recording_debug
from .client import InferenceClient
from .daemon import last_toggle_log_event, service_state
from .daemon_lifecycle import toggle_log_path
from .device import resolve_torch_device, torch_cuda_usable, torch_mps_available
from .gnome_shortcut import shortcut_readiness
from .models import (
    cache_artifacts_present,
    model_cache_root,
    normalize_asr_backend,
    required_model_cache_paths,
    resolve_catalog_entry,
)
from .paste import clipboard_capability, paste_capability
from .platform_capabilities import session_info
from .platform_runtime import PlatformRuntime, get_runtime
from .runtime_profile import detect_runtime_profile, is_apple_silicon, is_native_arm_python
from .settings import Settings, default_config_dir, settings_path


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


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


def build_backend_checks(settings: Settings, runtime: PlatformRuntime | None = None) -> list[Check]:
    if settings.asr_backend.startswith("file:"):
        return [
            check_model_cache(settings, runtime),
            Check("asr_runtime", True, "file backend has no extra runtime checks", required=True),
        ]
    try:
        backend = normalize_asr_backend(settings.asr_backend)
        entry = resolve_catalog_entry(settings, runtime)
    except ValueError as exc:
        return [Check("asr_config", False, str(exc))]
    checks = [check_model_cache(settings, runtime)]
    if entry is not None and entry.runtime_kind == "mlx":
        checks.extend(build_mlx_checks(settings, runtime))
    elif backend == "granite_nar":
        checks.extend(
            [
                check_torch_runtime(settings),
                check_asr_runtime(settings),
            ]
        )
    elif entry is not None and entry.runtime_kind == "torch":
        checks.append(check_torch_runtime(settings))
    else:
        checks.append(Check("asr_runtime", True, f"{settings.asr_backend} has no extra runtime checks"))
    return checks


def build_mlx_checks(settings: Settings, runtime: PlatformRuntime | None = None) -> list[Check]:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    checks = [
        Check(
            "mlx_platform",
            system == "Darwin",
            "MLX ASR requires macOS" if system != "Darwin" else "macOS",
        ),
        Check(
            "mlx_apple_silicon",
            is_apple_silicon(runtime),
            "MLX ASR requires Apple Silicon (arm64)"
            if not is_apple_silicon(runtime)
            else f"architecture={detect_runtime_profile(runtime).architecture}",
        ),
        Check(
            "mlx_native_python",
            is_native_arm_python(runtime),
            f"Python architecture is {py_platform.machine()}; MLX requires native ARM Python",
        ),
        check_macos_version(),
        check_mlx_import(),
        check_mlx_audio_import(),
        check_mlx_model_cache(settings, runtime),
    ]
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


def check_hotkey_readiness(
    settings: Settings | None = None,
    runtime: PlatformRuntime | None = None,
) -> Check:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() == "Darwin":
        return Check(
            "hotkey_readiness",
            True,
            "configure a macOS Keyboard Shortcut or Shortcuts.app action for transclip toggle-record --paste",
            required=False,
        )
    readiness = shortcut_readiness(
        expected_binding=(settings or Settings()).hotkey_linux,
        runtime=runtime,
    )
    return Check("hotkey_readiness", readiness.ok, readiness.detail)


def check_microphone_devices(
    settings: Settings | None = None,
    runtime: PlatformRuntime | None = None,
) -> Check:
    del settings
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    if system == "Darwin":
        try:
            import sounddevice as sd
        except ImportError:
            return Check(
                "microphone_devices",
                False,
                "sounddevice is not installed; install transclip[audio]",
            )
        try:
            default = sd.query_devices(kind="input")
            name = default.get("name", "unknown")
            return Check(
                "microphone_devices",
                True,
                f"default input: {name}; grant Microphone permission when prompted on first recording",
            )
        except Exception as exc:
            detail = str(exc)
            if "permission" in detail.lower() or "denied" in detail.lower():
                return Check(
                    "microphone_devices",
                    False,
                    "Microphone permission denied; open System Settings > Privacy & Security > Microphone",
                )
            return Check("microphone_devices", False, f"sounddevice input query failed: {detail}")
    if system != "Linux":
        return Check("microphone_devices", True, f"not checked on {system}")

    arecord = platform_runtime.which("arecord")
    if arecord:
        result = platform_runtime.run(
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

    if platform_runtime.which("wpctl"):
        result = platform_runtime.run(
            ["wpctl", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode == 0 and "Sources:" in result.stdout:
            return Check("microphone_devices", True, "wpctl reports audio sources")
    return Check("microphone_devices", False, "requires arecord or wpctl to inspect microphone devices")


def check_model_cache(settings: Settings, runtime: PlatformRuntime | None = None) -> Check:
    cache_root = model_cache_root(settings, runtime)
    required_paths = required_model_cache_paths(settings, runtime)
    if not required_paths:
        return Check("model_cache", True, "no local model artifacts required")
    missing = [str(path) for path in required_paths if not path.exists()]
    if not missing and cache_artifacts_present(settings.asr_model, settings, runtime):
        return Check("model_cache", True, f"found model artifacts under {cache_root}")
    if not missing:
        return Check("model_cache", True, f"found model artifacts under {cache_root}")
    return Check(
        "model_cache",
        False,
        "missing local model artifacts: "
        + ", ".join(missing)
        + f"; run: transclip models prefetch --model {settings.asr_model}",
    )


def check_mlx_model_cache(settings: Settings, runtime: PlatformRuntime | None = None) -> Check:
    if cache_artifacts_present(settings.asr_model, settings, runtime):
        return Check("mlx_model_cache", True, f"found MLX snapshot for {settings.asr_model}")
    return Check(
        "mlx_model_cache",
        False,
        f"missing MLX model cache for {settings.asr_model}; "
        f"run: transclip models prefetch --model {settings.asr_model}",
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
        return Check("torch_runtime", False, "torch is not installed; install transclip[models]")
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
    if normalize_asr_backend(settings.asr_backend) != "granite_nar":
        return Check("asr_runtime", True, f"{settings.asr_backend} has no extra runtime checks")
    try:
        import os

        import torch
    except ImportError:
        return Check("asr_runtime", False, "torch is not installed")
    device = resolve_torch_device(settings.asr_device)
    if device != "cuda":
        return Check("asr_runtime", True, f"Granite NAR will use {device} without flash-attn")
    if getattr(torch.version, "hip", None):
        os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        return Check(
            "asr_runtime",
            False,
            f"Granite NAR requires flash-attn (project-validated ROCm where available); import failed: {exc}",
        )
    return Check(
        "asr_runtime",
        True,
        "Granite NAR flash-attn runtime import passed (ROCm support is project-validated, not an IBM guarantee)",
    )


def check_mlx_import() -> Check:
    try:
        import mlx  # noqa: F401
    except ImportError as exc:
        return Check("mlx_import", False, f"mlx import failed: {exc}; install transclip[mlx]")
    return Check("mlx_import", True, "mlx import passed")


def check_mlx_audio_import() -> Check:
    try:
        import mlx_audio  # noqa: F401
    except ImportError as exc:
        return Check("mlx_audio_import", False, f"mlx_audio import failed: {exc}; install transclip[mlx]")
    return Check("mlx_audio_import", True, "mlx_audio import passed")


def check_macos_version() -> Check:
    if sys.platform != "darwin":
        return Check("macos_version", True, "not checked off macOS")
    version = tuple(int(part) for part in py_platform.mac_ver()[0].split(".")[:2] if part.isdigit())
    ok = version >= (14, 0) if version else False
    return Check(
        "macos_version",
        ok,
        f"macOS {py_platform.mac_ver()[0]}; MLX requires macOS >= 14.0",
    )


def check_tcc_permissions(runtime: PlatformRuntime | None = None) -> Check:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() != "Darwin":
        return Check("tcc_permissions", True, "not checked off macOS", required=False)
    return Check(
        "tcc_permissions",
        True,
        "verify manually: Microphone (recording), Accessibility and/or Automation (osascript paste). "
        "Permissions attach to Terminal, LaunchAgent Python, Shortcuts, or a packaged .app separately.",
        required=False,
    )


def checks_as_json(checks: list[Check]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2)


def checks_as_text(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        status = "ok" if check.ok else "missing"
        prefix = "" if check.required else "info\t"
        lines.append(f"{prefix}{status}\t{check.name}\t{check.detail}")
    return "\n".join(lines)
