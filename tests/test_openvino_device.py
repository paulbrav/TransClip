from __future__ import annotations

import unittest
from unittest.mock import patch

from transclip.openvino_device import has_intel_accelerator, resolve_openvino_device


class OpenVINODeviceTests(unittest.TestCase):
    @staticmethod
    def _devices(devices):
        return patch(
            "transclip.openvino_device.openvino_available_devices",
            return_value=tuple(devices),
        )

    def test_auto_resolves_to_auto_plugin(self):
        with self._devices(["CPU", "GPU.0", "NPU"]):
            self.assertEqual(resolve_openvino_device("auto"), "AUTO")
            self.assertEqual(resolve_openvino_device("openvino:AUTO"), "AUTO")
            self.assertEqual(resolve_openvino_device(""), "AUTO")

    def test_explicit_gpu_when_present(self):
        with self._devices(["CPU", "GPU.0"]):
            self.assertEqual(resolve_openvino_device("openvino:GPU"), "GPU")
            self.assertEqual(resolve_openvino_device("openvino:GPU.0"), "GPU.0")

    def test_npu_requested_but_absent_raises(self):
        with self._devices(["CPU", "GPU.0"]), self.assertRaisesRegex(RuntimeError, "NPU"):
            resolve_openvino_device("openvino:NPU")

    def test_cpu_always_allowed_even_without_devices(self):
        with self._devices([]):
            self.assertEqual(resolve_openvino_device("openvino:CPU"), "CPU")

    def test_has_intel_accelerator_detects_gpu_or_npu(self):
        with self._devices(["CPU", "GPU.0", "NPU"]):
            self.assertTrue(has_intel_accelerator())
        with self._devices(["CPU", "NPU"]):
            self.assertTrue(has_intel_accelerator())
        with self._devices(["CPU"]):
            self.assertFalse(has_intel_accelerator())
        with self._devices([]):
            self.assertFalse(has_intel_accelerator())

    def test_unknown_device_raises_value_error(self):
        with self._devices(["CPU"]), self.assertRaises(ValueError):
            resolve_openvino_device("openvino:TPU")


if __name__ == "__main__":
    unittest.main()
