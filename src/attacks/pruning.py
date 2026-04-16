"""剪枝攻击实现。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class PruningAttack(BaseAttack):
    """通过剪枝移除参数信息。"""

    def attack(
        self,
        model: nn.Module,
        pruning_rate: float = 0.3,
        method: str = "magnitude",
    ) -> nn.Module:
        if pruning_rate < 0 or pruning_rate >= 1:
            raise ValueError("pruning_rate must be in [0, 1)")

        with torch.no_grad():
            for param in model.parameters():
                if not torch.is_floating_point(param.data):
                    continue
                flat = param.data.view(-1)
                k = int(flat.numel() * pruning_rate)
                if k <= 0:
                    continue
                if method == "magnitude":
                    threshold = torch.kthvalue(flat.abs(), k).values
                    mask = flat.abs() > threshold
                elif method == "random":
                    idx = torch.randperm(flat.numel(), device=flat.device)
                    mask = torch.ones_like(flat, dtype=torch.bool)
                    mask[idx[:k]] = False
                else:
                    raise ValueError(f"Unknown pruning method: {method}")
                flat.mul_(mask.to(flat.dtype))
        return model

    def get_attack_name(self) -> str:
        return "pruning"
