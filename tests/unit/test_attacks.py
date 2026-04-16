"""攻击模块单元测试。"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.attacks.base_attack import BaseAttack
from src.attacks.fine_tuning import FineTuningAttack
from src.attacks.pruning import PruningAttack
from src.attacks.quantization import QuantizationAttack
from src.attacks.overwriting import OverwritingAttack
from src.attacks.ambiguity import AmbiguityAttack
from src.attacks.model_extraction import ModelExtractionAttack


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


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1).cpu() for p in model.parameters()])


class TestBaseAttack(unittest.TestCase):
    def test_base_attack_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseAttack()


class TestAttackImplementations(unittest.TestCase):
    def test_fine_tuning_attack_updates_model(self) -> None:
        model = TinyClassifier()
        before = flatten_params(model)
        attack = FineTuningAttack(device="cpu")
        attacked = attack.attack(model, train_loader=make_loader(), epochs=1, lr=0.01)
        after = flatten_params(attacked)
        self.assertFalse(torch.allclose(before, after))

    def test_pruning_attack_zeros_some_weights(self) -> None:
        model = TinyClassifier()
        attack = PruningAttack(device="cpu")
        attacked = attack.attack(model, pruning_rate=0.3, method="magnitude")
        params = flatten_params(attacked)
        self.assertGreater(torch.count_nonzero(params == 0).item(), 0)

    def test_quantization_attack_changes_parameter_grid(self) -> None:
        model = TinyClassifier()
        before = flatten_params(model)
        attack = QuantizationAttack(device="cpu")
        attacked = attack.attack(model, num_bits=4)
        after = flatten_params(attacked)
        self.assertFalse(torch.allclose(before, after))

    def test_overwriting_attack_modifies_model(self) -> None:
        model = TinyClassifier()
        before = flatten_params(model)
        attack = OverwritingAttack(device="cpu")
        attacked = attack.attack(model, strength=0.02)
        after = flatten_params(attacked)
        self.assertFalse(torch.allclose(before, after))

    def test_ambiguity_attack_generates_fake_fingerprint(self) -> None:
        attack = AmbiguityAttack(device="cpu", seed=7)
        target = torch.sign(torch.randn(32))
        fake = attack.generate_fake_fingerprint(target)
        self.assertEqual(fake.shape, target.shape)
        self.assertTrue(torch.all((fake == -1) | (fake == 1)))

    def test_model_extraction_attack_returns_surrogate(self) -> None:
        victim = TinyClassifier()
        surrogate = TinyClassifier()
        attack = ModelExtractionAttack(device="cpu")
        extracted = attack.attack(
            victim_model=victim,
            query_loader=make_loader(),
            surrogate_model=surrogate,
            epochs=1,
            lr=0.01,
        )
        self.assertIsInstance(extracted, nn.Module)

    def test_model_extraction_accepts_base_signature_model_kwarg(self) -> None:
        victim = TinyClassifier()
        attack = ModelExtractionAttack(device="cpu")
        extracted = attack.attack(
            model=victim,
            query_loader=make_loader(),
            epochs=1,
            lr=0.01,
        )
        self.assertIsInstance(extracted, nn.Module)


if __name__ == "__main__":
    unittest.main()
