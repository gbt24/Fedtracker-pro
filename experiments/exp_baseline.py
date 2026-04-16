"""基线对比实验脚本。

本文件属于 FedTracker-Pro 项目
功能: 提供基线实验默认攻击集合
依赖: src.attacks

代码生成来源: implementation_plan.md
章节: 第五阶段 实验系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import List

from src.attacks import FineTuningAttack, PruningAttack, QuantizationAttack


def build_default_attacks(device: str = "cuda") -> List:
    """构建基线实验默认攻击列表。"""
    return [
        FineTuningAttack(device=device),
        PruningAttack(device=device),
        QuantizationAttack(device=device),
    ]
