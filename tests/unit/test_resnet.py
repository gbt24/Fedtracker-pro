"""ResNet 模块单元测试。"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.resnet import BasicBlock, ResNet18, ResNet34


class TestBasicBlock(unittest.TestCase):
    """测试 BasicBlock 的形状行为。"""

    def test_basic_block_same_shape_when_stride_one(self) -> None:
        block = BasicBlock(in_planes=64, planes=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))

    def test_basic_block_downsample_when_stride_two(self) -> None:
        block = BasicBlock(in_planes=64, planes=128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 16, 16))


class TestResNetFactories(unittest.TestCase):
    """测试 ResNet 工厂函数输出维度。"""

    def test_resnet18_forward_shape(self) -> None:
        model = ResNet18(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_resnet34_forward_shape(self) -> None:
        model = ResNet34(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 100))

    def test_resnet18_supports_single_channel_input(self) -> None:
        model = ResNet18(num_classes=10, input_channels=1)
        x = torch.randn(3, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (3, 10))


if __name__ == "__main__":
    unittest.main()
