from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Literal

from transclip.platform.runtime import PlatformRuntime, get_runtime
from transclip.product import CONFIG_DIR_NAME, LOG_DIR_NAME

ProfileRuntimeKind = Literal["torch_cuda", "torch_rocm", "torch_mps", "torch_cpu", "mlx", "file"]
ProfileId = Literal[
    "linux_gpu",
    "linux_cpu",
    "darwin_arm_mlx",
    "darwin_other",
    "windows_cuda",
    "windows_cpu",
    "unsupported",
]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    profile_id: ProfileId
    system: str
    architecture: str
    default_asr_backend: str
    default_asr_model: str
    default_asr_device: str
    supported_runtime_kinds: tuple[ProfileRuntimeKind, ...]
    service_manager: Literal["systemd", "launchd", "task_scheduler", "none"]
    granite_nar_unsupported_reason: str | None = None
    config_dir_name: str = CONFIG_DIR_NAME
    log_dir_name: str = LOG_DIR_NAME


GRANITE_NAR_UNSUPPORTED_MACOS = (
    "Granite Speech 4.1 NAR is not supported on macOS. "
    'Set asr_backend = "mlx_audio_whisper" or choose a supported MLX model.'
)
GRANITE_NAR_UNSUPPORTED_WINDOWS = (
    "Granite Speech 4.1 NAR is not supported on Windows. "
    'Set asr_backend = "granite" with ibm-granite/granite-speech-4.1-2b.'
)


def machine_architecture(runtime: PlatformRuntime | None = None) -> str:
    platform_runtime_instance = get_runtime(runtime)
    if platform_runtime_instance.system() == "Darwin":
        try:
            output = platform_runtime_instance.check_output(["uname", "-m"])
            if isinstance(output, bytes):
                output = output.decode()
            return output.strip().lower()
        except Exception:
            pass
    return platform.machine().lower()


def is_apple_silicon(runtime: PlatformRuntime | None = None) -> bool:
    return machine_architecture(runtime) in {"arm64", "aarch64"}


def is_native_arm_python() -> bool:
    return platform.machine().lower() in {"arm64", "aarch64"}


def detect_runtime_profile(runtime: PlatformRuntime | None = None) -> RuntimeProfile:
    from transclip.device import torch_cuda_usable

    platform_runtime_instance = get_runtime(runtime)
    system = platform_runtime_instance.system()
    arch = machine_architecture(platform_runtime_instance)

    if system == "Linux":
        if arch in {"x86_64", "amd64"}:
            return RuntimeProfile(
                profile_id="linux_gpu",
                system=system,
                architecture=arch,
                default_asr_backend="granite_nar",
                default_asr_model="ibm-granite/granite-speech-4.1-2b-nar",
                default_asr_device="auto",
                supported_runtime_kinds=("torch_cuda", "torch_rocm", "torch_cpu", "file"),
                service_manager="systemd",
            )
        return RuntimeProfile(
            profile_id="linux_cpu",
            system=system,
            architecture=arch,
            default_asr_backend="granite",
            default_asr_model="ibm-granite/granite-speech-4.1-2b",
            default_asr_device="cpu",
            supported_runtime_kinds=("torch_cpu", "file"),
            service_manager="systemd",
        )

    if system == "Darwin":
        if arch in {"arm64", "aarch64"}:
            return RuntimeProfile(
                profile_id="darwin_arm_mlx",
                system=system,
                architecture=arch,
                default_asr_backend="mlx_audio_whisper",
                default_asr_model="mlx-community/whisper-large-v3-turbo-asr-fp16",
                default_asr_device="auto",
                supported_runtime_kinds=("mlx", "torch_mps", "torch_cpu", "file"),
                service_manager="launchd",
                granite_nar_unsupported_reason=GRANITE_NAR_UNSUPPORTED_MACOS,
            )
        return RuntimeProfile(
            profile_id="darwin_other",
            system=system,
            architecture=arch,
            default_asr_backend="file:/dev/null",
            default_asr_model="",
            default_asr_device="cpu",
            supported_runtime_kinds=("file",),
            service_manager="launchd",
            granite_nar_unsupported_reason=GRANITE_NAR_UNSUPPORTED_MACOS,
        )

    if system == "Windows":
        if torch_cuda_usable():
            return RuntimeProfile(
                profile_id="windows_cuda",
                system=system,
                architecture=arch,
                default_asr_backend="granite",
                default_asr_model="ibm-granite/granite-speech-4.1-2b",
                default_asr_device="auto",
                supported_runtime_kinds=("torch_cuda", "torch_cpu", "file"),
                service_manager="task_scheduler",
                granite_nar_unsupported_reason=GRANITE_NAR_UNSUPPORTED_WINDOWS,
            )
        return RuntimeProfile(
            profile_id="windows_cpu",
            system=system,
            architecture=arch,
            default_asr_backend="granite",
            default_asr_model="ibm-granite/granite-speech-4.1-2b",
            default_asr_device="cpu",
            supported_runtime_kinds=("torch_cpu", "file"),
            service_manager="task_scheduler",
            granite_nar_unsupported_reason=GRANITE_NAR_UNSUPPORTED_WINDOWS,
        )

    return RuntimeProfile(
        profile_id="unsupported",
        system=system,
        architecture=arch,
        default_asr_backend="file:/dev/null",
        default_asr_model="",
        default_asr_device="cpu",
        supported_runtime_kinds=("file",),
        service_manager="none",
    )
