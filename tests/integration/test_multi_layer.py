"""三层验证集成测试。"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defense.multi_layer_verify import MultiLayerVerifier


class _WatermarkStub:
    def verify(self, model: nn.Module) -> float:
        _ = model
        return 0.95


class _FingerprintStub:
    def verify(self, model: nn.Module) -> float:
        _ = model
        return 0.9


def test_multi_layer_verify_passes_all_levels() -> None:
    model = nn.Linear(4, 2)
    verifier = MultiLayerVerifier(_WatermarkStub(), _FingerprintStub())

    result = verifier.verify_ownership(model=model, crypto_result={"is_valid": True})

    assert result["is_verified"] is True
    assert result["level1"]["passed"] is True
    assert result["level2"]["passed"] is True
    assert result["level3"]["passed"] is True
