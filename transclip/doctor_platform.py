from __future__ import annotations

import platform as py_platform
import subprocess

from .doctor_types import Check
from .gnome_shortcut import shortcut_readiness
from .hotkey_setup import macos_hotkey_setup_message
from .platform_runtime import PlatformRuntime, get_runtime
from .settings import Settings, active_hotkey


def _check_sounddevice_input(*, permission_hint: str, denied_hint: str) -> Check:
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
        return Check("microphone_devices", True, f"default input: {name}; {permission_hint}")
    except Exception as exc:
        detail = str(exc)
        if "permission" in detail.lower() or "denied" in detail.lower():
            return Check("microphone_devices", False, denied_hint)
        return Check("microphone_devices", False, f"sounddevice input query failed: {detail}")


def check_hotkey_readiness(
    settings: Settings | None = None,
    runtime: PlatformRuntime | None = None,
) -> Check:
    platform_runtime = get_runtime(runtime)
    current = settings or Settings()
    if platform_runtime.system() == "Darwin":
        return Check(
            "hotkey_readiness",
            True,
            macos_hotkey_setup_message(current, runtime=platform_runtime),
            required=False,
        )
    if platform_runtime.system() == "Windows":
        binding = active_hotkey(current, platform_runtime)
        return Check(
            "hotkey_readiness",
            True,
            f"configured hotkey {binding!r}; global hotkey is registered when transclip tray is running",
            required=False,
        )
    readiness = shortcut_readiness(
        expected_binding=current.hotkey_linux,
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
    if system in {"Darwin", "Windows"}:
        hints = {
            "Darwin": (
                "grant Microphone permission when prompted on first recording",
                "Microphone permission denied; open System Settings > Privacy & Security > Microphone",
            ),
            "Windows": (
                "enable Microphone access in Windows Settings > Privacy > Microphone",
                "Microphone permission denied; open Settings > Privacy & security > Microphone",
            ),
        }
        permission_hint, denied_hint = hints[system]
        return _check_sounddevice_input(permission_hint=permission_hint, denied_hint=denied_hint)
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


def check_macos_version(runtime: PlatformRuntime | None = None) -> Check:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() != "Darwin":
        return Check("macos_version", True, "not checked off macOS")
    version = tuple(int(part) for part in py_platform.mac_ver()[0].split(".")[:2] if part.isdigit())
    ok = version >= (14, 0) if version else False
    return Check(
        "macos_version",
        ok,
        f"macOS {py_platform.mac_ver()[0]}; MLX requires macOS >= 14.0",
    )


def check_windows_version(runtime: PlatformRuntime | None = None) -> Check:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() != "Windows":
        return Check("windows_version", True, "not checked off Windows")
    release = py_platform.release()
    try:
        major = int(release.split(".")[0])
    except ValueError:
        major = 0
    ok = major >= 10
    return Check(
        "windows_version",
        ok,
        f"Windows {py_platform.version()}; Windows 10+ recommended",
        required=False,
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
