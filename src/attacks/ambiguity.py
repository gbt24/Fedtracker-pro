"""模糊攻击实现。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class AmbiguityAttack(BaseAttack):
    """通过生成假指纹制造归属歧义。"""

    def generate_fake_fingerprint(
        self, target_fingerprint: torch.Tensor
    ) -> torch.Tensor:
        if target_fingerprint.numel() == 0:
            raise ValueError("target_fingerprint must be non-empty")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        fake = (
            torch.randint(0, 2, target_fingerprint.shape, generator=generator).float()
            * 2
            - 1
        )
        return fake.to(target_fingerprint.device)

    def attack(
        self,
        model: nn.Module,
        target_fingerprint: torch.Tensor | None = None,
        fake_strength: float = 0.01,
    ) -> nn.Module:
        if fake_strength <= 0:
            raise ValueError("fake_strength must be greater than 0")

        if target_fingerprint is None:
            target_fingerprint = torch.ones(128)
        fake = self.generate_fake_fingerprint(target_fingerprint).view(-1)

        idx = 0
        with torch.no_grad():
            for param in model.parameters():
                if not torch.is_floating_point(param.data):
                    continue
                flat = param.data.view(-1)
                seg_len = min(len(flat), len(fake))
                segment = fake[idx : idx + seg_len]
                if len(segment) < seg_len:
                    segment = fake[:seg_len]
                repeated = segment.repeat((len(flat) + seg_len - 1) // seg_len)[
                    : len(flat)
                ]
                flat.add_(repeated.to(flat.device) * fake_strength)
                idx = (idx + seg_len) % len(fake)
        return model

    def get_attack_name(self) -> str:
        return "ambiguity"
