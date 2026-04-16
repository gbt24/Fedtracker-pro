"""指纹模块单元测试。"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defense.fingerprint.base_fingerprint import BaseFingerprint
from src.defense.fingerprint.param_fingerprint import ParametricFingerprint


class TinyMLP(nn.Module):
    """测试用小型 MLP。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TestBaseFingerprint(unittest.TestCase):
    """测试指纹抽象基类。"""

    def test_base_fingerprint_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseFingerprint()


class TestParametricFingerprint(unittest.TestCase):
    """测试参数指纹功能。"""

    def test_generate_returns_expected_dimension(self) -> None:
        fp = ParametricFingerprint(fingerprint_dim=64, seed=7, device="cpu")
        code = fp.generate()
        self.assertEqual(code.shape, (64,))
        self.assertTrue(torch.all((code == -1) | (code == 1)))

    def test_embed_and_extract_shapes(self) -> None:
        model = TinyMLP()
        fp = ParametricFingerprint(
            fingerprint_dim=32, embedding_strength=0.01, device="cpu"
        )
        fp.generate()
        fp.embed(model)
        extracted = fp.extract(model)
        self.assertEqual(extracted.shape, (32,))
        self.assertTrue(torch.all((extracted == -1) | (extracted == 1)))

    def test_verify_returns_similarity_in_range(self) -> None:
        model = TinyMLP()
        fp = ParametricFingerprint(
            fingerprint_dim=32, embedding_strength=0.01, device="cpu"
        )
        fp.generate()
        fp.embed(model)
        score = fp.verify(model)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_embedded_score_is_higher_than_clean_score(self) -> None:
        clean_model = TinyMLP()
        embedded_model = TinyMLP()
        fp = ParametricFingerprint(
            fingerprint_dim=32,
            embedding_strength=0.02,
            device="cpu",
            seed=123,
        )
        fp.generate()

        clean_score = fp.verify(clean_model)
        fp.embed(embedded_model)
        embedded_score = fp.verify(embedded_model)
        self.assertGreater(embedded_score, clean_score)

    def test_verify_requires_generated_fingerprint(self) -> None:
        model = TinyMLP()
        fp = ParametricFingerprint(fingerprint_dim=16, device="cpu")
        with self.assertRaises(ValueError):
            fp.verify(model)

    def test_invalid_fingerprint_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            ParametricFingerprint(fingerprint_dim=0, device="cpu")

    def test_embed_only_changes_limited_parameter_positions(self) -> None:
        model = TinyMLP()
        before = torch.cat([p.detach().view(-1).cpu() for p in model.parameters()])

        fp = ParametricFingerprint(
            fingerprint_dim=16,
            embedding_strength=0.02,
            device="cpu",
            seed=9,
        )
        fp.generate()
        fp.embed(model)

        after = torch.cat([p.detach().view(-1).cpu() for p in model.parameters()])
        changed = torch.count_nonzero((after - before).abs() > 1e-12).item()
        self.assertLessEqual(changed, 16)


if __name__ == "__main__":
    unittest.main()
