"""量化攻击实现。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class QuantizationAttack(BaseAttack):
    """通过降低参数精度削弱鲁棒性。"""

    def attack(self, model: nn.Module, num_bits: int = 8) -> nn.Module:
        if num_bits <= 0 or num_bits > 16:
            raise ValueError("num_bits must be in (0, 16]")

        levels = float(2**num_bits - 1)
        with torch.no_grad():
            for param in model.parameters():
                if not torch.is_floating_point(param.data):
                    continue
                min_val = param.data.min()
                max_val = param.data.max()
                if torch.isclose(max_val, min_val):
                    continue
                normalized = (param.data - min_val) / (max_val - min_val)
                quantized = torch.round(normalized * levels) / levels
                param.data.copy_(quantized * (max_val - min_val) + min_val)
        return model

    def get_attack_name(self) -> str:
        return "quantization"
