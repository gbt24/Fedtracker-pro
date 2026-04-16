"""FedTrackerPro 主框架单元测试。"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.attacks.fine_tuning import FineTuningAttack


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
    x = torch.randn(24, 3, 8, 8)
    y = torch.randint(0, 4, (24,), dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


class DummyDataManager:
    def __init__(self, num_clients: int) -> None:
        self.num_clients = num_clients
        self._loaders = [make_loader() for _ in range(num_clients)]
        self._test_loader = make_loader()

    def get_client_loader(
        self, client_id: int, batch_size: int = 8, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        _ = batch_size, shuffle
        return self._loaders[client_id]

    def get_test_loader(self, batch_size: int = 8) -> torch.utils.data.DataLoader:
        _ = batch_size
        return self._test_loader


class NoOpAttack:
    def attack(self, model: nn.Module) -> nn.Module:
        return model

    def get_attack_name(self) -> str:
        return "noop"


class TestFedTrackerPro(unittest.TestCase):
    def _build_config(self) -> Config:
        config = Config()
        config.system.device = "cpu"
        config.system.save_frequency = 1
        config.federated.num_clients = 2
        config.federated.client_fraction = 1.0
        config.federated.local_epochs = 1
        config.federated.local_lr = 0.01
        config.federated.local_batch_size = 8
        config.federated.optimizer = "sgd"
        config.watermark.enabled = False
        config.fingerprint.enabled = False
        config.adaptive.enabled = False
        config.crypto.enabled = False
        config.unlearning.enabled = False
        return config

    def test_initialize_creates_server_and_clients(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        self.assertIsNotNone(framework.server)
        self.assertEqual(len(framework.clients), 2)

    def test_train_one_round_updates_server_round(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        framework.train(num_rounds=1)

        self.assertEqual(framework.server.round_num, 1)

    def test_verify_ownership_without_defense_modules(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        is_owner, leaker_id, confidence = framework.verify_ownership(TinyClassifier())

        self.assertTrue(is_owner)
        self.assertEqual(leaker_id, 0)
        self.assertAlmostEqual(confidence, 1.0, places=6)

    def test_verify_ownership_rejects_on_crypto_failure(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        class _CryptoFailStub:
            def verify_model(self, model: nn.Module) -> dict[str, bool]:
                _ = model
                return {"is_valid": False}

        framework.crypto_verifier = _CryptoFailStub()
        is_owner, leaker_id, _ = framework.verify_ownership(TinyClassifier())

        self.assertFalse(is_owner)
        self.assertIsNone(leaker_id)

    def test_verify_ownership_rejects_on_watermark_failure(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        class _WatermarkFailStub:
            def verify(self, model: nn.Module) -> float:
                _ = model
                return 0.0

        framework.watermarker = _WatermarkFailStub()
        is_owner, _, _ = framework.verify_ownership(TinyClassifier())

        self.assertFalse(is_owner)

    def test_evaluate_attack_robustness_returns_named_results(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        results = framework.evaluate_attack_robustness([NoOpAttack()], make_loader())

        self.assertIn("noop", results)
        self.assertEqual(results["noop"], 1.0)

    def test_evaluate_attack_robustness_supports_fine_tuning_attack(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        results = framework.evaluate_attack_robustness(
            [FineTuningAttack(device="cpu")],
            make_loader(),
        )

        self.assertIn("fine_tuning", results)


if __name__ == "__main__":
    unittest.main()
