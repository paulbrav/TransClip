from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import platform
import shutil
import subprocess

from .device import torch_cuda_usable, torch_mps_available
from .settings import Settings, default_config_dir, keywords_path, settings_path


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_checks(settings: Settings, config_dir: Path | None = None) -> list[Check]:
    return [
        check_config_files(config_dir),
        check_clipboard_tools(),
        check_paste_tools(),
        check_microphone_devices(),
        check_tauri_linux_libs(),
        check_model_cache(settings),
        check_torch_runtime(settings),
        check_asr_runtime(settings),
    ]


def check_config_files(config_dir: Path | None = None) -> Check:
    missing = [
        str(path)
        for path in (settings_path(config_dir), keywords_path(config_dir))
        if not path.exists()
    ]
    if not missing:
        return Check("config_files", True, f"found files in {config_dir or default_config_dir()}")
    return Check("config_files", False, "missing: " + ", ".join(missing))


def check_clipboard_tools() -> Check:
    system = platform.system()
    if system == "Darwin":
        return Check("clipboard_tools", bool(shutil.which("pbcopy") and shutil.which("pbpaste")), "requires pbcopy and pbpaste")
    tools = [tool for tool in ("wl-copy", "wl-paste", "xclip", "xsel") if shutil.which(tool)]
    return Check(
        "clipboard_tools",
        bool(tools),
        "found: " + ", ".join(tools) if tools else "requires wl-clipboard, xclip, or xsel; apt: wl-clipboard xclip",
    )


def check_paste_tools() -> Check:
    system = platform.system()
    if system == "Darwin":
        return Check("paste_tools", bool(shutil.which("osascript")), "requires osascript and Accessibility permission")
    session = (os_environ("XDG_SESSION_TYPE") or "").lower()
    wtype = shutil.which("wtype")
    xdotool = shutil.which("xdotool")
    ydotool = shutil.which("ydotool")
    if session == "wayland":
        details = []
        if wtype:
            result = subprocess.run(
                [wtype, "-M", "ctrl", "-m", "ctrl"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return Check("paste_tools", True, "wtype found and compositor accepts virtual keyboard events")
            detail = result.stdout.strip() or f"exit status {result.returncode}"
            details.append(f"wtype unusable: {detail}")
        if ydotool:
            return Check(
                "paste_tools",
                True,
                "ydotool found; requires ydotoold/uinput permissions to inject paste",
            )
        if not details:
            return Check(
                "paste_tools",
                False,
                "Wayland paste injection requires wtype or ydotool; apt: wtype ydotool",
            )
        return Check(
            "paste_tools",
            False,
            "; ".join(details) + "; fallback requires ydotool with ydotoold/uinput permissions",
        )
    if xdotool:
        return Check("paste_tools", True, "found: xdotool")
    if ydotool:
        return Check("paste_tools", True, "found: ydotool")
    if wtype:
        return Check("paste_tools", False, "wtype is installed, but non-Wayland sessions require xdotool or ydotool; apt: xdotool ydotool")
    return Check("paste_tools", False, "requires wtype, xdotool, or ydotool; apt: xdotool ydotool")


def check_microphone_devices() -> Check:
    system = platform.system()
    if system == "Darwin":
        return Check(
            "microphone_devices",
            True,
            "macOS microphone visibility is checked by the Tauri/WebAudio permission prompt",
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
            devices = [
                line.strip()
                for line in output.splitlines()
                if line.strip().startswith("card ")
            ]
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


def check_tauri_linux_libs() -> Check:
    if platform.system() != "Linux":
        return Check("tauri_linux_libs", True, "not a Linux host")
    missing = [
        package
        for package in ("libsoup-3.0", "javascriptcoregtk-4.1", "webkit2gtk-4.1")
        if not pkg_config_exists(package)
    ]
    if not missing:
        return Check("tauri_linux_libs", True, "pkg-config found Tauri WebKitGTK libraries")
    apt = (
        "apt: libwebkit2gtk-4.1-dev build-essential curl wget file "
        "libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev"
    )
    return Check("tauri_linux_libs", False, "missing pkg-config packages: " + ", ".join(missing) + "; " + apt)


def check_model_cache(settings: Settings) -> Check:
    cache_root = Path(settings.model_cache_dir).expanduser() if settings.model_cache_dir else Path.home() / ".cache" / "huggingface" / "hub"
    required_paths = [cache_root / hf_cache_dir(settings.asr_model)]
    if settings.cleanup_runtime == "llama_cpp":
        required_paths.append(Path(settings.cleanup_model_path).expanduser())
    elif settings.cleanup_runtime == "transformers":
        required_paths.append(cache_root / hf_cache_dir(settings.cleanup_model))
    missing = [str(path) for path in required_paths if not path.exists()]
    if not missing:
        return Check("model_cache", True, f"found model artifacts under {cache_root}")
    return Check("model_cache", False, "missing local model artifacts: " + ", ".join(missing))


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


def hf_cache_dir(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def pkg_config_exists(package: str) -> bool:
    return subprocess.run(
        ["pkg-config", "--exists", package],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


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
