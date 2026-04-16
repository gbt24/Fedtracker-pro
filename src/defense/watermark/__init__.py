"""水印模块导出。

本文件属于 FedTracker-Pro 项目
功能: 导出水印基类与持续学习水印实现
依赖: src.defense.watermark.base_watermark, src.defense.watermark.cl_watermark

代码生成来源: code_generation_guide.md
章节: 阶段4 水印系统
生成日期: 2026-04-16
"""

from .base_watermark import BaseWatermark
from .cl_watermark import ContinualLearningWatermark

__all__ = ["BaseWatermark", "ContinualLearningWatermark"]
