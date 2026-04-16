"""防御模块导出。

本文件属于 FedTracker-Pro 项目
功能: 导出水印与指纹防御组件
依赖: src.defense.watermark, src.defense.fingerprint

代码生成来源: code_generation_guide.md
章节: 阶段4 防御模块
生成日期: 2026-04-16
"""

from .watermark import BaseWatermark, ContinualLearningWatermark
from .fingerprint import BaseFingerprint, ParametricFingerprint

__all__ = [
    "BaseWatermark",
    "ContinualLearningWatermark",
    "BaseFingerprint",
    "ParametricFingerprint",
]
