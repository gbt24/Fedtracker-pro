"""三层验证模块单元测试。"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defense.multi_layer_verify import MultiLayerVerifier


class DummyWatermark:
    def __init__(self, score: float) -> None:
        self.score = score

    def verify(self, model) -> float:
        _ = model
        return self.score


class DummyFingerprint:
    def __init__(self, score: float) -> None:
        self.score = score

    def verify(self, model) -> float:
        _ = model
        return self.score


class TestMultiLayerVerifier(unittest.TestCase):
    """测试三层验证聚合结果。"""

    def test_verify_ownership_returns_layered_results(self) -> None:
        verifier = MultiLayerVerifier(
            watermark=DummyWatermark(0.92),
            fingerprint=DummyFingerprint(0.88),
            level1_threshold=0.75,
            level3_threshold=0.9,
        )
        model = torch.nn.Linear(4, 2)
        result = verifier.verify_ownership(model, crypto_result=True)
        self.assertTrue(result["is_verified"])
        self.assertIn("level1", result)
        self.assertIn("level2", result)
        self.assertIn("level3", result)

    def test_verify_ownership_fails_when_crypto_fails(self) -> None:
        verifier = MultiLayerVerifier(
            watermark=DummyWatermark(0.95),
            fingerprint=DummyFingerprint(0.95),
        )
        model = torch.nn.Linear(4, 2)
        result = verifier.verify_ownership(model, crypto_result=False)
        self.assertFalse(result["is_verified"])

    def test_verify_ownership_accepts_crypto_dict(self) -> None:
        verifier = MultiLayerVerifier(
            watermark=DummyWatermark(0.95),
            fingerprint=DummyFingerprint(0.95),
        )
        model = torch.nn.Linear(4, 2)
        result = verifier.verify_ownership(model, crypto_result={"is_valid": True})
        self.assertTrue(result["level2"]["passed"])


if __name__ == "__main__":
    unittest.main()
