from pathlib import Path

from tests.service_helpers import FakeRuntime


def linux_runtime(home: Path | None = None) -> FakeRuntime:
    return FakeRuntime(system="Linux", home=home or Path("/home/test"))


def darwin_arm_runtime(home: Path | None = None) -> FakeRuntime:
    return FakeRuntime(system="Darwin", home=home or Path("/Users/test"), check_output_text="arm64")
