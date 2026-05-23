from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Literal

from .platform_runtime import PlatformRuntime, get_runtime, user_cache_dir, user_config_dir, user_log_dir
from .product import CONFIG_DIR_NAME, LOG_DIR_NAME

RuntimeKind = Literal["torch_cuda", "torch_rocm", "torch_mps", "torch_cpu", "mlx", "file"]
ProfileId = Literal["linux_gpu", "linux_cpu", "darwin_arm_mlx", "darwin_other", "unsupported"]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    profile_id: ProfileId
    system: str
    architecture: str
    default_asr_backend: str
    default_asr_model: str
    default_asr_device: str
    supported_runtime_kinds: tuple[RuntimeKind, ...]
    service_manager: Literal["systemd", "launchd", "none"]
    config_dir_name: str = CONFIG_DIR_NAME
    log_dir_name: str = LOG_DIR_NAME


def machine_architecture(runtime: PlatformRuntime | None = None) -> str:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() == "Darwin":
        try:
            output = platform_runtime.check_output(["uname", "-m"])
            if isinstance(output, bytes):
                output = output.decode()
            return output.strip().lower()
        except Exception:
            pass
    return platform.machine().lower()


def is_apple_silicon(runtime: PlatformRuntime | None = None) -> bool:
    return machine_architecture(runtime) in {"arm64", "aarch64"}


def is_native_arm_python(runtime: PlatformRuntime | None = None) -> bool:
    return machine_architecture(runtime) in {"arm64", "aarch64"}


def detect_runtime_profile(runtime: PlatformRuntime | None = None) -> RuntimeProfile:
    platform_runtime = get_runtime(runtime)
    system = platform_runtime.system()
    arch = machine_architecture(platform_runtime)

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


def profile_config_dir(profile: RuntimeProfile, runtime: PlatformRuntime | None = None) -> str:
    del profile
    return str(user_config_dir(CONFIG_DIR_NAME, runtime))


def profile_cache_dir(profile: RuntimeProfile, runtime: PlatformRuntime | None = None) -> str:
    del profile
    return str(user_cache_dir(CONFIG_DIR_NAME, runtime))


def profile_log_dir(profile: RuntimeProfile, runtime: PlatformRuntime | None = None) -> str:
    del profile
    return str(user_log_dir(LOG_DIR_NAME, runtime))


def profile_model_cache_hint(runtime: PlatformRuntime | None = None) -> str:
    return str(user_cache_dir("huggingface", runtime) / "hub")
