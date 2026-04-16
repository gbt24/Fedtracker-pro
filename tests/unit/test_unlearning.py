"""遗忘增强模块单元测试。"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defense.unlearning_guided import UnlearningGuidedRelocation


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loader() -> torch.utils.data.DataLoader:
    x = torch.randn(20, 3, 8, 8)
    y = torch.randint(0, 4, (20,), dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=5, shuffle=False)


class TestUnlearningGuidedRelocation(unittest.TestCase):
    """测试稳定参数识别与指纹重定位。"""

    def test_identify_stable_parameters_returns_scores(self) -> None:
        relocator = UnlearningGuidedRelocation(simulation_steps=1, device="cpu")
        scores = relocator.identify_stable_parameters(
            TinyClassifier(), make_loader(), sample_ratio=0.5
        )
        self.assertTrue(scores)
        self.assertTrue(all(0.0 < value <= 1.0 for value in scores.values()))

    def test_relocate_fingerprint_preserves_vector_length(self) -> None:
        model = TinyClassifier()
        relocator = UnlearningGuidedRelocation(simulation_steps=1, device="cpu")
        fingerprint = torch.randn(32)
        updated = relocator.relocate_fingerprint(
            model, fingerprint, make_loader(), strength=0.01
        )
        self.assertIs(updated, model)

    def test_relocate_fingerprint_rejects_empty_vector(self) -> None:
        model = TinyClassifier()
        relocator = UnlearningGuidedRelocation(simulation_steps=1, device="cpu")
        with self.assertRaises(ValueError):
            relocator.relocate_fingerprint(
                model, torch.tensor([]), make_loader(), strength=0.01
            )


if __name__ == "__main__":
    unittest.main()
