"""VGG 与 MobileNet 模块单元测试。"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.mobilenet import MobileNetV2
from src.models.vgg import VGG11, VGG16


class TestVGGFactories(unittest.TestCase):
    """测试 VGG 模型工厂函数。"""

    def test_vgg11_forward_shape(self) -> None:
        model = VGG11(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_vgg16_forward_shape(self) -> None:
        model = VGG16(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 100))


class TestMobileNetFactory(unittest.TestCase):
    """测试 MobileNetV2 工厂函数。"""

    def test_mobilenetv2_forward_shape(self) -> None:
        model = MobileNetV2(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_mobilenetv2_supports_single_channel(self) -> None:
        model = MobileNetV2(num_classes=10, input_channels=1)
        x = torch.randn(3, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (3, 10))


if __name__ == "__main__":
    unittest.main()
