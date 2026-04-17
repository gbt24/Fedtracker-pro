"""FedTrackerPro 主框架单元测试。"""

import os
import sys
import unittest
from unittest.mock import patch

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

    def _build_config_with_fingerprint(self, num_clients: int = 3) -> Config:
        config = Config()
        config.system.device = "cpu"
        config.system.save_frequency = 1
        config.federated.num_clients = num_clients
        config.federated.client_fraction = 1.0
        config.federated.local_epochs = 1
        config.federated.local_lr = 0.01
        config.federated.local_batch_size = 8
        config.federated.optimizer = "sgd"
        config.fingerprint.enabled = True
        config.fingerprint.fingerprint_dim = 32
        config.fingerprint.embedding_strength = 0.02
        config.fingerprint.min_strength = 0.01
        config.fingerprint.identification_threshold = 0.5
        config.watermark.enabled = False
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

    def test_train_fetches_client_loader_on_demand(self) -> None:
        data_manager = DummyDataManager(num_clients=2)
        with patch.object(
            data_manager,
            "get_client_loader",
            wraps=data_manager.get_client_loader,
        ) as mocked_get_loader:
            framework = FedTrackerPro(self._build_config())
            framework.initialize(TinyClassifier(), data_manager=data_manager)

            self.assertEqual(mocked_get_loader.call_count, 0)
            framework.train(num_rounds=1)

        self.assertEqual(mocked_get_loader.call_count, 2)

    def test_initialize_passes_loader_settings_to_data_manager(self) -> None:
        config = self._build_config()
        config.system.num_workers = 3
        config.system.persistent_workers = True
        config.system.prefetch_factor = 3

        with patch(
            "src.core.fed_tracker_pro.FederatedDataManager",
            return_value=DummyDataManager(num_clients=2),
        ) as mock_manager:
            framework = FedTrackerPro(config)
            framework.initialize(TinyClassifier())

        mock_manager.assert_called_once_with(
            dataset_name=config.data.dataset,
            data_dir=config.data.data_dir,
            num_clients=config.federated.num_clients,
            iid=config.data.iid,
            alpha=config.data.alpha,
            num_shards=config.data.num_shards,
            num_workers=3,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=3,
        )

    def test_train_one_round_updates_server_round(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        framework.train(num_rounds=1)

        self.assertEqual(framework.server.round_num, 1)

    def test_train_supports_progress_mode(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        framework.train(num_rounds=1, show_progress=True, progress_desc="test-train")

        self.assertEqual(framework.server.round_num, 1)

    def test_verify_ownership_without_defense_modules(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        is_owner, leaker_id, confidence = framework.verify_ownership(TinyClassifier())

        self.assertTrue(is_owner)
        self.assertIsNotNone(leaker_id)
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

    def test_evaluate_attack_robustness_supports_progress_mode(self) -> None:
        framework = FedTrackerPro(self._build_config())
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        results = framework.evaluate_attack_robustness(
            [NoOpAttack()],
            make_loader(),
            show_progress=True,
            progress_desc="test-attacks",
        )

        self.assertIn("noop", results)

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

    def test_initialize_with_adaptive_allocator_enabled(self) -> None:
        config = self._build_config()
        config.adaptive.enabled = True
        config.adaptive.beta = 0.2
        config.adaptive.min_allocation = 0.03
        config.adaptive.evaluation_period = 7

        framework = FedTrackerPro(config)
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        self.assertIsNotNone(framework.adaptive_allocator)
        self.assertAlmostEqual(framework.adaptive_allocator.beta, 0.2)
        self.assertAlmostEqual(framework.adaptive_allocator.min_allocation, 0.03)
        self.assertEqual(framework.adaptive_allocator.evaluation_period, 7)
        self.assertEqual(framework.adaptive_allocator.device, framework.device)


class TestFedTrackerProPerClientTracing(unittest.TestCase):
    """测试逐客户端溯源功能。"""

    def test_fingerprint_registry_initialized(self) -> None:
        config = self._build_config_with_fingerprint(num_clients=3)
        framework = FedTrackerPro(config)
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=3)
        )

        self.assertIsNotNone(framework.fingerprint_registry)
        self.assertEqual(framework.fingerprint_registry.registered_ids, [0, 1, 2])

    def test_per_client_fingerprint_identifies_correct_leaker(self) -> None:
        config = self._build_config_with_fingerprint(num_clients=5)
        framework = FedTrackerPro(config)
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=5)
        )

        framework.train(num_rounds=1)

        victim_model = framework.clients[3].model
        is_owner, leaker_id, confidence = framework.verify_ownership(victim_model)

        self.assertTrue(is_owner)
        self.assertEqual(leaker_id, 3)
        self.assertGreater(confidence, 0.5)

    def test_per_client_fingerprint_rejects_unknown_model(self) -> None:
        config = self._build_config_with_fingerprint(num_clients=3)
        framework = FedTrackerPro(config)
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=3)
        )
        framework.train(num_rounds=1)

        unknown_model = TinyClassifier()
        is_owner, leaker_id, _ = framework.verify_ownership(unknown_model)

        self.assertFalse(is_owner)
        self.assertIsNone(leaker_id)

    def test_cross_client_low_false_positive(self) -> None:
        config = self._build_config_with_fingerprint(num_clients=5)
        framework = FedTrackerPro(config)
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=5)
        )
        framework.train(num_rounds=1)

        registry = framework.fingerprint_registry
        model_0 = framework.clients[0].model
        sims = registry.get_all_similarities(model_0)

        score_0 = sims[0]
        for cid in range(1, 5):
            self.assertLess(
                sims[cid],
                score_0,
                f"Client {cid} similarity >= client 0 self-similarity",
            )

    def test_per_client_fingerprint_after_fine_tuning(self) -> None:
        config = self._build_config_with_fingerprint(num_clients=3)
        framework = FedTrackerPro(config)
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=3)
        )
        framework.train(num_rounds=1)

        import copy

        victim_model = copy.deepcopy(framework.clients[1].model)
        attack = FineTuningAttack(device="cpu")
        attacked_model = attack.attack(
            victim_model, train_loader=make_loader(), epochs=1, lr=0.001
        )

        is_owner, leaker_id, _ = framework.verify_ownership(attacked_model)
        self.assertEqual(leaker_id, 1)

    def test_clients_are_protected_type(self) -> None:
        from src.core.protected_client import ProtectedClient

        config = self._build_config_with_fingerprint(num_clients=2)
        framework = FedTrackerPro(config)
        framework.initialize(
            TinyClassifier(), data_manager=DummyDataManager(num_clients=2)
        )

        for client in framework.clients:
            self.assertIsInstance(client, ProtectedClient)

    def _build_config_with_fingerprint(self, num_clients: int = 3) -> Config:
        config = Config()
        config.system.device = "cpu"
        config.system.save_frequency = 1
        config.federated.num_clients = num_clients
        config.federated.client_fraction = 1.0
        config.federated.local_epochs = 1
        config.federated.local_lr = 0.01
        config.federated.local_batch_size = 8
        config.federated.optimizer = "sgd"
        config.fingerprint.enabled = True
        config.fingerprint.fingerprint_dim = 32
        config.fingerprint.embedding_strength = 0.02
        config.fingerprint.min_strength = 0.01
        config.fingerprint.identification_threshold = 0.5
        config.watermark.enabled = False
        config.adaptive.enabled = False
        config.crypto.enabled = False
        config.unlearning.enabled = False
        return config


if __name__ == "__main__":
    unittest.main()
