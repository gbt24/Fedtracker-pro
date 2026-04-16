"""联邦聚合模块导出。

本文件属于 FedTracker-Pro 项目
功能: 导出聚合器基类及 FedAvg/FedProx 实现
依赖: src.aggregation.base_aggregator, src.aggregation.fed_avg, src.aggregation.fed_prox

代码生成来源: code_generation_guide.md
章节: 阶段2 文件清单
生成日期: 2026-04-16
"""

from .base_aggregator import BaseAggregator
from .fed_avg import FedAvgAggregator
from .fed_prox import FedProxAggregator

__all__ = ["BaseAggregator", "FedAvgAggregator", "FedProxAggregator"]
