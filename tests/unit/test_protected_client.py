"""受保护客户端单元测试。"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.protected_client import ProtectedClient
from src.defense.fingerprint.param_fingerprint import ParametricFingerprint


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loader() -> torch.utils.data.DataLoader:
    x = torch.randn(16, 16)
    y = torch.randint(0, 10, (16,), dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=4)


class TestProtectedClient(unittest.TestCase):
    """测试受保护客户端。"""

    def test_embed_protection_modifies_model(self) -> None:
        model = TinyMLP()
        fp = ParametricFingerprint(
            fingerprint_dim=16, embedding_strength=0.02, device="cpu", seed=1
        )
        fp.generate()
        client = ProtectedClient(
            client_id=0,
            model=model,
            train_loader=make_loader(),
            fingerprinter=fp,
            device="cpu",
            local_epochs=1,
            local_lr=0.01,
        )
        before = {k: v.clone() for k, v in model.state_dict().items()}
        client.embed_protection()
        changed = False
        for k in before:
            if not torch.equal(before[k], model.state_dict()[k]):
                changed = True
                break
        self.assertTrue(changed)

    def test_embed_protection_fingerprint_verifiable(self) -> None:
        model = TinyMLP()
        fp = ParametricFingerprint(
            fingerprint_dim=16, embedding_strength=0.02, device="cpu", seed=5
        )
        fp.generate()
        client = ProtectedClient(
            client_id=0,
            model=model,
            train_loader=make_loader(),
            fingerprinter=fp,
            device="cpu",
            local_epochs=1,
            local_lr=0.01,
        )
        client.embed_protection()
        score = fp.verify(model)
        self.assertGreater(score, 0.75)

    def test_local_train_includes_protection(self) -> None:
        model = TinyMLP()
        fp = ParametricFingerprint(
            fingerprint_dim=16, embedding_strength=0.02, device="cpu", seed=9
        )
        fp.generate()
        client = ProtectedClient(
            client_id=0,
            model=model,
            train_loader=make_loader(),
            fingerprinter=fp,
            device="cpu",
            local_epochs=1,
            local_lr=0.01,
        )
        global_state = {k: v.clone() for k, v in model.state_dict().items()}
        client.local_train(global_state=global_state)
        score = fp.verify(client.model)
        self.assertGreater(score, 0.75)

    def test_different_clients_get_different_fingerprints(self) -> None:
        fp0 = ParametricFingerprint(
            fingerprint_dim=32, embedding_strength=0.02, device="cpu", seed=100
        )
        fp1 = ParametricFingerprint(
            fingerprint_dim=32, embedding_strength=0.02, device="cpu", seed=101
        )
        fp0.generate()
        fp1.generate()

        model0 = TinyMLP()
        model1 = TinyMLP()

        fp0.embed(model0)
        fp1.embed(model1)

        score_0_on_0 = fp0.verify(model0)
        score_1_on_0 = fp1.verify(model0)
        self.assertGreater(score_0_on_0, score_1_on_0)


if __name__ == "__main__":
    unittest.main()
