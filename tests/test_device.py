from __future__ import annotations

import unittest
from unittest.mock import patch

from granite_speach.device import resolve_torch_device


class DeviceTests(unittest.TestCase):
    def test_auto_uses_cpu_when_accelerators_are_unusable(self) -> None:
        with (
            patch("granite_speach.device.torch_cuda_usable", return_value=False),
            patch("granite_speach.device.torch_mps_available", return_value=False),
        ):
            self.assertEqual(resolve_torch_device("auto"), "cpu")

    def test_rocm_request_requires_working_cuda_compatible_torch(self) -> None:
        with (
            patch("granite_speach.device.torch_cuda_usable", return_value=False),
            self.assertRaisesRegex(RuntimeError, "CUDA/ROCm was requested"),
        ):
            resolve_torch_device("rocm")

    def test_auto_uses_cuda_when_gpu_smoke_passes(self) -> None:
        with patch("granite_speach.device.torch_cuda_usable", return_value=True):
            self.assertEqual(resolve_torch_device("auto"), "cuda")


if __name__ == "__main__":
    unittest.main()
