"""自适应水印分配模块。

本文件属于 FedTracker-Pro 项目
功能: 根据客户端容忍度动态分配水印预算
依赖: math

代码生成来源: implementation_plan.md
章节: 阶段5 自适应分配系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Dict, Optional


class AdaptiveAllocator:
    """根据客户端表现动态分配预算。"""

    def __init__(
        self,
        beta: float = 0.1,
        min_allocation: float = 0.05,
        evaluation_period: int = 10,
        device: Optional[str] = None,
    ) -> None:
        if beta <= 0:
            raise ValueError("beta must be greater than 0")
        if min_allocation < 0:
            raise ValueError("min_allocation must be non-negative")
        if evaluation_period <= 0:
            raise ValueError("evaluation_period must be greater than 0")
        self.beta = beta
        self.min_allocation = min_allocation
        self.evaluation_period = evaluation_period
        self.device = device

    def evaluate_tolerance(
        self,
        accuracy: float,
        loss: float,
        fingerprint_similarity: float,
    ) -> float:
        """估计客户端对保护扰动的容忍度。"""
        acc_term = max(0.0, min(accuracy, 1.0))
        loss_term = 1.0 / (1.0 + max(loss, 0.0))
        sim_term = max(0.0, min(fingerprint_similarity, 1.0))
        score = 0.4 * acc_term + 0.3 * loss_term + 0.3 * sim_term
        return max(0.0, min(score, 1.0))

    def allocate(self, tolerance_scores: Dict[str, float]) -> Dict[str, float]:
        """根据容忍度分配总预算 beta。"""
        if not tolerance_scores:
            return {}

        clamped = {key: max(value, 0.0) for key, value in tolerance_scores.items()}
        total = sum(clamped.values())
        n_clients = len(clamped)

        if total == 0:
            uniform = self.beta / n_clients
            return {key: uniform for key in clamped}

        extra_budget = max(0.0, self.beta - self.min_allocation * n_clients)
        allocations = {
            key: self.min_allocation + extra_budget * (value / total)
            for key, value in clamped.items()
        }

        current_total = sum(allocations.values())
        if current_total != self.beta:
            scale = self.beta / current_total if current_total > 0 else 0.0
            allocations = {key: value * scale for key, value in allocations.items()}

        return allocations
