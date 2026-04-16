"""水印基类。

本文件属于 FedTracker-Pro 项目
功能: 定义水印系统统一抽象接口
依赖: torch, torch.nn, abc

代码生成来源: code_generation_guide.md
章节: 阶段4 水印系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseWatermark(ABC):
    """水印基类。"""

    def __init__(
        self,
        trigger_size: int = 100,
        target_label: int = 0,
        device: str = "cuda",
    ) -> None:
        if trigger_size <= 0:
            raise ValueError("trigger_size must be greater than 0")
        if target_label < 0:
            raise ValueError("target_label must be non-negative")
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.trigger_set: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @abstractmethod
    def generate_trigger_set(
        self,
        data_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成触发集。"""

    @abstractmethod
    def embed(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 0.001,
    ) -> nn.Module:
        """嵌入水印。"""

    @abstractmethod
    def verify(self, model: nn.Module) -> float:
        """验证水印。"""
