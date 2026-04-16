"""指纹基类。

本文件属于 FedTracker-Pro 项目
功能: 定义指纹系统统一抽象接口
依赖: torch, torch.nn, abc

代码生成来源: code_generation_guide.md
章节: 阶段4 指纹系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFingerprint(ABC):
    """指纹基类。"""

    def __init__(
        self,
        fingerprint_dim: int = 128,
        embedding_strength: float = 0.1,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        if fingerprint_dim <= 0:
            raise ValueError("fingerprint_dim must be greater than 0")
        if embedding_strength <= 0:
            raise ValueError("embedding_strength must be greater than 0")
        self.fingerprint_dim = fingerprint_dim
        self.embedding_strength = embedding_strength
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.seed = seed
        self.fingerprint: torch.Tensor | None = None

    @abstractmethod
    def generate(self) -> torch.Tensor:
        """生成指纹码。"""

    @abstractmethod
    def embed(self, model: nn.Module) -> nn.Module:
        """嵌入指纹。"""

    @abstractmethod
    def extract(self, model: nn.Module) -> torch.Tensor:
        """提取指纹。"""

    @abstractmethod
    def verify(self, model: nn.Module) -> float:
        """验证指纹。"""
