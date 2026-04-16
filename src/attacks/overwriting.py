"""覆盖攻击实现。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class OverwritingAttack(BaseAttack):
    """通过写入新扰动覆盖原始保护痕迹。"""

    def attack(self, model: nn.Module, strength: float = 0.01) -> nn.Module:
        if strength <= 0:
            raise ValueError("strength must be greater than 0")
        generators: dict[str, torch.Generator] = {}
        with torch.no_grad():
            for param in model.parameters():
                if not torch.is_floating_point(param.data):
                    continue
                use_native_generator = param.device.type in {"cpu", "cuda"}
                key = (
                    str(param.device)
                    if use_native_generator
                    else f"cpu->{param.device}"
                )
                generator = generators.get(key)
                if generator is None:
                    generator_device: torch.device | str
                    if use_native_generator:
                        generator_device = param.device
                    else:
                        generator_device = "cpu"
                    generator = torch.Generator(device=generator_device)
                    generator.manual_seed(self.seed)
                    generators[key] = generator

                if use_native_generator:
                    noise = torch.randn(
                        param.shape,
                        generator=generator,
                        device=param.device,
                        dtype=param.dtype,
                    )
                else:
                    noise = torch.randn(
                        param.shape,
                        generator=generator,
                        device="cpu",
                        dtype=param.dtype,
                    ).to(param.device)
                param.add_(noise * strength)
        return model

    def get_attack_name(self) -> str:
        return "overwriting"
