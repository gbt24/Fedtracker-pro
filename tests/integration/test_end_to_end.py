"""端到端流程集成测试。"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro


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
    x = torch.randn(24, 3, 8, 8)
    y = torch.randint(0, 4, (24,), dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


class _DummyDataManager:
    def __init__(self, num_clients: int) -> None:
        self._loaders = [_make_loader() for _ in range(num_clients)]
        self._test = _make_loader()

    def get_client_loader(
        self, client_id: int, batch_size: int = 8, shuffle: bool = True
    ):
        _ = batch_size, shuffle
        return self._loaders[client_id]

    def get_test_loader(self, batch_size: int = 8):
        _ = batch_size
        return self._test


def test_end_to_end_one_round_runs() -> None:
    cfg = Config()
    cfg.system.device = "cpu"
    cfg.system.save_frequency = 1
    cfg.federated.num_clients = 2
    cfg.federated.client_fraction = 1.0
    cfg.federated.local_epochs = 1
    cfg.watermark.enabled = False
    cfg.fingerprint.enabled = False
    cfg.adaptive.enabled = False
    cfg.crypto.enabled = False
    cfg.unlearning.enabled = False

    framework = FedTrackerPro(cfg)
    framework.initialize(
        TinyClassifier(), data_manager=_DummyDataManager(num_clients=2)
    )
    framework.train(num_rounds=1)

    assert framework.server is not None
    assert framework.server.round_num == 1
