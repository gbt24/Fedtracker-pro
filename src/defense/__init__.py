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
from .adaptive_allocation import AdaptiveAllocator
from .crypto_verification import CryptographicVerification
from .unlearning_guided import UnlearningGuidedRelocation
from .multi_layer_verify import MultiLayerVerifier

__all__ = [
    "BaseWatermark",
    "ContinualLearningWatermark",
    "BaseFingerprint",
    "ParametricFingerprint",
    "AdaptiveAllocator",
    "CryptographicVerification",
    "UnlearningGuidedRelocation",
    "MultiLayerVerifier",
]
