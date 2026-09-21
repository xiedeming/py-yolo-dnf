import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.hardware_profile import (
    GIB,
    HardwareInfo,
    detect_hardware,
    select_runtime_profile,
)


class FakeCuda:
    def __init__(self, memories, available=True):
        self._memories = memories
        self._available = available

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self._memories)

    def get_device_properties(self, index):
        return SimpleNamespace(
            name=f"GPU-{index}",
            total_memory=self._memories[index],
        )


class HardwareProfileTests(unittest.TestCase):
    def test_exactly_four_gib_is_high_spec(self):
        hardware = detect_hardware(SimpleNamespace(cuda=FakeCuda([4 * GIB])))

        self.assertTrue(hardware.is_high_spec)
        self.assertEqual(hardware.best_gpu.index, 0)
        self.assertEqual(hardware.max_memory_bytes, 4 * GIB)

    def test_multiple_gpus_choose_largest_memory_and_lowest_index_on_tie(self):
        hardware = detect_hardware(SimpleNamespace(cuda=FakeCuda([6 * GIB, 8 * GIB, 8 * GIB])))

        self.assertTrue(hardware.is_high_spec)
        self.assertEqual(hardware.best_gpu.index, 1)

    def test_unavailable_cuda_is_low_spec(self):
        hardware = detect_hardware(SimpleNamespace(cuda=FakeCuda([16 * GIB], available=False)))

        self.assertFalse(hardware.cuda_available)
        self.assertFalse(hardware.is_high_spec)
        self.assertEqual(hardware.gpus, ())

    def test_profile_paths_and_manual_device_overrides(self):
        root = Path("D:/project")
        low_hardware = HardwareInfo(False, reason="CUDA 不可用")

        automatic = select_runtime_profile(str(root), hardware=low_hardware)
        manual_high = select_runtime_profile(str(root), requested_device="cuda", hardware=low_hardware)
        manual_low = select_runtime_profile(str(root), requested_device="cpu", hardware=automatic.hardware)

        self.assertEqual(automatic.name, "low")
        self.assertEqual(automatic.device, "cpu")
        self.assertEqual(automatic.config_path, root / "config" / "settings.cpu.yaml")
        self.assertEqual(manual_high.name, "high")
        self.assertEqual(manual_high.device, "cuda")
        self.assertEqual(manual_high.config_path, root / "config" / "settings.yaml")
        self.assertEqual(manual_low.name, "low")

    def test_manual_device_override_does_not_import_or_probe_cuda(self):
        with patch("src.utils.hardware_profile.detect_hardware") as probe:
            profile = select_runtime_profile(requested_device="cpu")

        probe.assert_not_called()
        self.assertEqual(profile.device, "cpu")


if __name__ == "__main__":
    unittest.main()
