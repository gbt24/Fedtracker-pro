"""攻击基类。

本文件属于 FedTracker-Pro 项目
功能: 定义攻击统一抽象接口
依赖: torch, abc

代码生成来源: implementation_plan.md
章节: 阶段6 攻击基类
生成日期: 2026-04-16
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAttack(ABC):
    """攻击基类。"""

    def __init__(self, device: str = "cuda", seed: int = 42) -> None:
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.seed = seed
        torch.manual_seed(seed)

    @abstractmethod
    def attack(self, model: nn.Module, **kwargs) -> nn.Module:
        """执行攻击。"""

    @abstractmethod
    def get_attack_name(self) -> str:
        """返回攻击名称。"""
