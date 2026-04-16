"""水印模块单元测试。"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defense.watermark.base_watermark import BaseWatermark
from src.defense.watermark.cl_watermark import ContinualLearningWatermark


class TinyClassifier(nn.Module):
    """测试用微型分类模型。"""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def make_loader(num_samples: int = 24, channels: int = 3):
    x = torch.randn(num_samples, channels, 32, 32)
    y = torch.randint(0, 10, (num_samples,), dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


class TestBaseWatermark(unittest.TestCase):
    """测试水印抽象基类。"""

    def test_base_watermark_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseWatermark()

    def test_invalid_trigger_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            ContinualLearningWatermark(trigger_size=0, device="cpu")


class TestContinualLearningWatermark(unittest.TestCase):
    """测试持续学习水印实现。"""

    def test_generate_trigger_set_shapes(self) -> None:
        wm = ContinualLearningWatermark(trigger_size=12, target_label=2, device="cpu")
        trigger_x, trigger_y = wm.generate_trigger_set(pattern_type="checkerboard")
        self.assertEqual(trigger_x.shape, (12, 3, 32, 32))
        self.assertEqual(trigger_y.shape, (12,))
        self.assertTrue(torch.all(trigger_y == 2))

    def test_generate_trigger_set_rejects_unknown_pattern(self) -> None:
        wm = ContinualLearningWatermark(trigger_size=4, device="cpu")
        with self.assertRaises(ValueError):
            wm.generate_trigger_set(pattern_type="unknown-pattern")

    def test_generate_trigger_set_rejects_empty_loader(self) -> None:
        wm = ContinualLearningWatermark(trigger_size=4, device="cpu")
        empty_ds = torch.utils.data.TensorDataset(
            torch.empty(0, 3, 32, 32),
            torch.empty(0, dtype=torch.long),
        )
        empty_loader = torch.utils.data.DataLoader(empty_ds, batch_size=4)
        with self.assertRaises(ValueError):
            wm.generate_trigger_set(data_loader=empty_loader)

    def test_verify_requires_trigger_set(self) -> None:
        wm = ContinualLearningWatermark(trigger_size=4, device="cpu")
        model = TinyClassifier()
        with self.assertRaises(ValueError):
            wm.verify(model)

    def test_embed_and_verify_roundtrip(self) -> None:
        model = TinyClassifier()
        loader = make_loader()
        wm = ContinualLearningWatermark(
            trigger_size=8,
            target_label=1,
            device="cpu",
            memory_size=4,
        )
        wm.embed(model, train_loader=loader, epochs=1, lr=0.001)
        score = wm.verify(model)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
