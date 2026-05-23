from __future__ import annotations

import platform as py_platform
import subprocess

from .doctor_types import Check
from .gnome_shortcut import shortcut_readiness
from .hotkey_setup import macos_hotkey_setup_message
from .platform_runtime import PlatformRuntime, get_runtime
from .settings import Settings


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
