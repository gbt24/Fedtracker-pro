"""FedProx 聚合算法。

本文件属于 FedTracker-Pro 项目
功能: 实现带近端正则项的联邦聚合
依赖: torch, src.aggregation.base_aggregator

代码生成来源: code_generation_guide.md
章节: 阶段2 FedProx
生成日期: 2026-04-16
"""

from typing import Dict, List, Optional

import torch

from .base_aggregator import BaseAggregator


class FedProxAggregator(BaseAggregator):
    """FedProx 聚合器。"""

    def __init__(self, device: str = "cuda", mu: float = 0.01) -> None:
        super().__init__(device)
        self.mu = mu

    def aggregate(
        self,
        local_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """执行 FedProx 聚合。"""
        if not local_states:
            raise ValueError("No local states to aggregate")
        if global_state is None:
            raise ValueError("global_state is required for FedProx aggregation")

        if weights is None:
            weights = [1.0 / len(local_states)] * len(local_states)
        elif len(weights) != len(local_states):
            raise ValueError("weights length must match local_states length")
        else:
            if any(weight < 0 for weight in weights):
                raise ValueError("weights must be non-negative")
            weight_sum = sum(weights)
            if weight_sum <= 0:
                raise ValueError("weights sum must be positive")
            weights = [weight / weight_sum for weight in weights]

        aggregated_state: Dict[str, torch.Tensor] = {}

        for key in global_state.keys():
            first_param = local_states[0][key]
            if not torch.is_floating_point(first_param):
                aggregated_state[key] = first_param.detach().cpu().clone()
                continue

            weighted_sum = sum(
                weight * local_state[key].to(self.device)
                for local_state, weight in zip(local_states, weights)
            )
            prox_term = self.mu * sum(
                weight
                * (global_state[key].to(self.device) - local_state[key].to(self.device))
                for local_state, weight in zip(local_states, weights)
            )
            aggregated_state[key] = (weighted_sum + prox_term).cpu()

        return aggregated_state
