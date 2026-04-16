"""参数指纹实现。

本文件属于 FedTracker-Pro 项目
功能: 生成、嵌入、提取并验证模型参数指纹
依赖: torch, torch.nn, src.defense.fingerprint.base_fingerprint

代码生成来源: code_generation_guide.md
章节: 阶段4 指纹系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .base_fingerprint import BaseFingerprint


class ParametricFingerprint(BaseFingerprint):
    """参数级指纹实现。"""

    def __init__(
        self,
        fingerprint_dim: int = 128,
        embedding_strength: float = 0.1,
        min_strength: float = 0.05,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        if min_strength <= 0:
            raise ValueError("min_strength must be greater than 0")
        super().__init__(fingerprint_dim, embedding_strength, device, seed)
        self.min_strength = min_strength
        self.embedding_indices: List[Tuple[str, int]] = []

    def generate(self) -> torch.Tensor:
        """生成 {-1, 1} 指纹码。"""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        bits = torch.randint(0, 2, (self.fingerprint_dim,), generator=generator)
        self.fingerprint = bits.float() * 2 - 1
        return self.fingerprint.clone()

    def _collect_flat_params(self, model: nn.Module) -> tuple[List[str], torch.Tensor]:
        """收集模型浮点参数并展平。"""
        names: List[str] = []
        flat_parts: List[torch.Tensor] = []
        for name, param in model.named_parameters():
            if torch.is_floating_point(param.data):
                names.append(name)
                flat_parts.append(param.data.detach().view(-1).cpu())
        if not flat_parts:
            raise ValueError("No floating-point parameters found in model")
        return names, torch.cat(flat_parts)

    def _indices_from_seed(self, total_size: int) -> torch.Tensor:
        """根据随机种子生成嵌入索引。"""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        perm = torch.randperm(total_size, generator=generator)
        return perm[: self.fingerprint_dim]

    def embed(self, model: nn.Module) -> nn.Module:
        """将指纹嵌入模型参数。"""
        if self.fingerprint is None:
            self.generate()

        _, flat = self._collect_flat_params(model)
        if self.fingerprint_dim > flat.numel():
            raise ValueError("fingerprint_dim exceeds number of model parameters")

        indices = self._indices_from_seed(flat.numel())
        strength = max(self.embedding_strength, self.min_strength)
        updated = flat.clone()
        selected = updated[indices]
        quantized = torch.round(selected / strength).to(torch.int64)
        bits = (self.fingerprint > 0).to(torch.int64)
        quantized = (quantized & ~1) | bits  # type: ignore[index]
        updated[indices] = quantized.float() * strength

        offset = 0
        for _, param in model.named_parameters():
            if not torch.is_floating_point(param.data):
                continue
            numel = param.numel()
            new_values = updated[offset : offset + numel].view_as(param.data)
            param.data.copy_(new_values.to(param.data.device))
            offset += numel

        self.embedding_indices = [("flat", int(i.item())) for i in indices]
        return model

    def extract(self, model: nn.Module) -> torch.Tensor:
        """从模型参数提取指纹。"""
        _, flat = self._collect_flat_params(model)
        if self.fingerprint_dim > flat.numel():
            raise ValueError("fingerprint_dim exceeds number of model parameters")

        indices = self._indices_from_seed(flat.numel())
        strength = max(self.embedding_strength, self.min_strength)
        quantized = torch.round(flat / strength).to(torch.int64)
        bits = quantized[indices] & 1
        return bits.float() * 2 - 1

    def verify(self, model: nn.Module) -> float:
        """验证提取指纹与原始指纹的余弦相似度。"""
        if self.fingerprint is None:
            raise ValueError("Fingerprint not generated")

        extracted = self.extract(model)
        score = torch.nn.functional.cosine_similarity(
            extracted.view(1, -1),
            self.fingerprint.view(1, -1),
            dim=1,
        )
        return float(score.item())
