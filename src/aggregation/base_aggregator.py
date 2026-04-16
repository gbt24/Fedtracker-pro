"""联邦聚合器基类。

本文件属于 FedTracker-Pro 项目
功能: 定义联邦聚合统一抽象接口
依赖: torch, abc

代码生成来源: code_generation_guide.md
章节: 阶段2 聚合器基类
生成日期: 2026-04-16
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch


class BaseAggregator(ABC):
    """联邦聚合器基类。"""

    def __init__(self, device: str = "cuda") -> None:
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )

    @abstractmethod
    def aggregate(
        self,
        local_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """执行模型聚合。"""

    def _compute_weights(
        self,
        local_states: List[Dict[str, torch.Tensor]],
        dataset_sizes: Optional[List[int]] = None,
    ) -> List[float]:
        """计算聚合权重。"""
        if dataset_sizes is not None:
            if len(dataset_sizes) != len(local_states):
                raise ValueError("dataset_sizes length must match local_states length")
            if any(size < 0 for size in dataset_sizes):
                raise ValueError("dataset_sizes cannot contain negative values")
            total = sum(dataset_sizes)
            if total <= 0:
                raise ValueError("Total dataset size must be positive")
            return [size / total for size in dataset_sizes]

        n_clients = len(local_states)
        if n_clients == 0:
            raise ValueError("local_states cannot be empty")
        return [1.0 / n_clients] * n_clients
