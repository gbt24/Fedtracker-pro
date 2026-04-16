"""Diffusion 模型单元测试。"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.diffusion import DiffusionUNet


class TestDiffusionUNet(unittest.TestCase):
    def test_forward_preserves_image_shape(self) -> None:
        model = DiffusionUNet(in_channels=3, base_channels=16)
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,), dtype=torch.long)
        out = model(x, t)
        self.assertEqual(out.shape, x.shape)

    def test_supports_single_channel(self) -> None:
        model = DiffusionUNet(in_channels=1, base_channels=8)
        x = torch.randn(1, 1, 28, 28)
        t = torch.randint(0, 1000, (1,), dtype=torch.long)
        out = model(x, t)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
