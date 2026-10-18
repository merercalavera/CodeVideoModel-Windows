import unittest

from cvm.devices import select_text_device


class DeviceSelectionTests(unittest.TestCase):
    def test_single_gpu_uses_generation_device(self):
        self.assertEqual(select_text_device("auto", "cuda:0", 1), "cuda:0")

    def test_multi_gpu_preserves_upstream_default(self):
        self.assertEqual(select_text_device("auto", "cuda:0", 2), "cuda:1")

    def test_explicit_device_is_preserved(self):
        self.assertEqual(select_text_device("cuda:0", "cuda:1", 2), "cuda:0")

    def test_no_gpu_does_not_invent_second_gpu(self):
        self.assertEqual(select_text_device("auto", "cuda:0", 0), "cuda:0")


if __name__ == "__main__":
    unittest.main()
