"""鲁棒性实验脚本。

本文件属于 FedTracker-Pro 项目
功能: 提供鲁棒性实验使用的攻击集合
依赖: src.attacks

代码生成来源: code_generation_guide.md
章节: 阶段8 实验脚本
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import List

from src.attacks import (
    AmbiguityAttack,
    FineTuningAttack,
    ModelExtractionAttack,
    OverwritingAttack,
    PruningAttack,
    QuantizationAttack,
)


def build_robustness_attacks(device: str = "cuda") -> List:
    """构建鲁棒性实验攻击列表。"""
    return [
        FineTuningAttack(device=device),
        PruningAttack(device=device),
        QuantizationAttack(device=device),
        OverwritingAttack(device=device),
        AmbiguityAttack(device=device),
        ModelExtractionAttack(device=device),
    ]
