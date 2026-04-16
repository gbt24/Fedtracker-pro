"""FedAvg 聚合算法。

本文件属于 FedTracker-Pro 项目
功能: 实现标准联邦平均聚合
依赖: torch, src.aggregation.base_aggregator

代码生成来源: code_generation_guide.md
章节: 阶段2 FedAvg
生成日期: 2026-04-16
"""

from typing import Dict, List, Optional

import torch

from .base_aggregator import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """FedAvg 聚合器。"""

    def aggregate(
        self,
        local_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
        dataset_sizes: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """执行 FedAvg 聚合。"""
        _ = global_state
        if not local_states:
            raise ValueError("No local states to aggregate")

        if weights is None:
            weights = self._compute_weights(local_states, dataset_sizes)
        elif len(weights) != len(local_states):
            raise ValueError("weights length must match local_states length")
        else:
            if any(weight < 0 for weight in weights):
                raise ValueError("weights must be non-negative")
            weight_sum = sum(weights)
            if weight_sum <= 0:
                raise ValueError("weights sum must be positive")
            weights = [weight / weight_sum for weight in weights]

        keys = local_states[0].keys()
        global_state: Dict[str, torch.Tensor] = {}

        for key in keys:
            first_param = local_states[0][key]
            if not torch.is_floating_point(first_param):
                global_state[key] = first_param.detach().cpu().clone()
                continue

            weighted_sum: Optional[torch.Tensor] = None
            for local_state, weight in zip(local_states, weights):
                param = local_state[key].to(self.device)
                if weighted_sum is None:
                    weighted_sum = weight * param
                else:
                    weighted_sum += weight * param
            if weighted_sum is None:
                raise ValueError(f"Failed to aggregate parameter: {key}")
            global_state[key] = weighted_sum.cpu()

        return global_state
