"""微调攻击实现。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_attack import BaseAttack


class FineTuningAttack(BaseAttack):
    """通过继续训练削弱保护信号。"""

    def attack(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 3,
        lr: float = 0.001,
    ) -> nn.Module:
        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            for data, target in train_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        return model

    def get_attack_name(self) -> str:
        return "fine_tuning"
