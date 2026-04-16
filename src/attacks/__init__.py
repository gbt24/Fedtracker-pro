"""攻击模块导出。

本文件属于 FedTracker-Pro 项目
功能: 导出攻击基类及各类攻击实现
依赖: src.attacks.*

代码生成来源: implementation_plan.md
章节: 阶段6 攻击模块
生成日期: 2026-04-16
"""

from .base_attack import BaseAttack
from .fine_tuning import FineTuningAttack
from .pruning import PruningAttack
from .quantization import QuantizationAttack
from .overwriting import OverwritingAttack
from .ambiguity import AmbiguityAttack
from .model_extraction import ModelExtractionAttack

__all__ = [
    "BaseAttack",
    "FineTuningAttack",
    "PruningAttack",
    "QuantizationAttack",
    "OverwritingAttack",
    "AmbiguityAttack",
    "ModelExtractionAttack",
]
