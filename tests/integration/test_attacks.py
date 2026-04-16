"""攻击模块集成测试。"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.attacks import (
    AmbiguityAttack,
    FineTuningAttack,
    ModelExtractionAttack,
    OverwritingAttack,
    PruningAttack,
    QuantizationAttack,
)


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


def _make_loader() -> torch.utils.data.DataLoader:
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 4, (16,), dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


def test_attack_pipeline_smoke() -> None:
    model = TinyClassifier()
    loader = _make_loader()

    model = FineTuningAttack(device="cpu").attack(model, train_loader=loader, epochs=1)
    model = PruningAttack(device="cpu").attack(model, pruning_rate=0.2)
    model = QuantizationAttack(device="cpu").attack(model, num_bits=8)
    model = OverwritingAttack(device="cpu").attack(model, strength=0.01)
    model = AmbiguityAttack(device="cpu").attack(model)
    extracted = ModelExtractionAttack(device="cpu").attack(
        model=model,
        query_loader=loader,
        epochs=1,
    )

    assert isinstance(extracted, nn.Module)
