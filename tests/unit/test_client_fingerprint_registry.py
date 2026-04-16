"""客户端指纹注册表单元测试。"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defense.fingerprint.client_fingerprint_registry import (
    ClientFingerprintRegistry,
)


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


class TestClientFingerprintRegistry(unittest.TestCase):
    """测试客户端指纹注册表。"""

    def test_register_client_creates_unique_fingerprints(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=64, device="cpu", base_seed=42
        )
        fp0 = registry.register_client(0)
        fp1 = registry.register_client(1)
        fp2 = registry.register_client(2)
        self.assertFalse(torch.equal(fp0.fingerprint, fp1.fingerprint))
        self.assertFalse(torch.equal(fp1.fingerprint, fp2.fingerprint))

    def test_identify_client_returns_correct_id(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=32, embedding_strength=0.02, device="cpu", base_seed=10
        )
        for i in range(5):
            registry.register_client(i)

        model = TinyMLP()
        registry.embed_client_fingerprint(3, model)

        matched_id, score = registry.identify_client(model)
        self.assertEqual(matched_id, 3)
        self.assertGreater(score, 0.8)

    def test_identify_client_returns_negative_on_threshold(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=32,
            embedding_strength=0.02,
            device="cpu",
            base_seed=10,
            identification_threshold=0.5,
        )
        for i in range(3):
            registry.register_client(i)

        clean_model = TinyMLP()
        matched_id, score = registry.identify_client(clean_model)
        self.assertEqual(matched_id, -1)

    def test_get_all_similarities_shape(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=16, device="cpu", base_seed=1
        )
        for i in range(4):
            registry.register_client(i)
        model = TinyMLP()
        sims = registry.get_all_similarities(model)
        self.assertEqual(len(sims), 4)
        for cid in range(4):
            self.assertIn(cid, sims)

    def test_different_clients_have_low_cross_similarity(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=64, embedding_strength=0.02, device="cpu", base_seed=42
        )
        for i in range(10):
            registry.register_client(i)

        model = TinyMLP()
        registry.embed_client_fingerprint(0, model)
        sims = registry.get_all_similarities(model)

        for cid in range(1, 10):
            self.assertLess(sims[cid], 0.5, f"Client {cid} cross-similarity too high")

    def test_fingerprint_survives_mild_perturbation(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=32, embedding_strength=0.02, device="cpu", base_seed=7
        )
        for i in range(3):
            registry.register_client(i)

        model = TinyMLP()
        registry.embed_client_fingerprint(1, model)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.001)

        matched_id, score = registry.identify_client(model)
        self.assertEqual(matched_id, 1)

    def test_register_duplicate_client_raises(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=16, device="cpu", base_seed=0
        )
        registry.register_client(0)
        with self.assertRaises(ValueError):
            registry.register_client(0)

    def test_get_fingerprint_raises_on_unknown(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=16, device="cpu", base_seed=0
        )
        with self.assertRaises(KeyError):
            registry.get_fingerprint(99)

    def test_identify_with_candidates_filter(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=32, embedding_strength=0.02, device="cpu", base_seed=5
        )
        for i in range(5):
            registry.register_client(i)

        model = TinyMLP()
        registry.embed_client_fingerprint(2, model)

        matched_id, _ = registry.identify_client(model, candidate_ids=[1, 2, 3])
        self.assertEqual(matched_id, 2)

    def test_register_clients_batch(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=16, device="cpu", base_seed=0
        )
        registry.register_clients([0, 1, 2])
        self.assertEqual(registry.registered_ids, [0, 1, 2])

    def test_identify_empty_registry(self) -> None:
        registry = ClientFingerprintRegistry(
            fingerprint_dim=16, device="cpu", base_seed=0
        )
        matched_id, score = registry.identify_client(TinyMLP())
        self.assertEqual(matched_id, -1)
        self.assertEqual(score, 0.0)

    def test_invalid_identification_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            ClientFingerprintRegistry(
                fingerprint_dim=16, device="cpu", identification_threshold=2.0
            )


if __name__ == "__main__":
    unittest.main()
