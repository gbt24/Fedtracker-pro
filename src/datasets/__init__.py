"""数据集模块导出。

本文件属于 FedTracker-Pro 项目
功能: 导出联邦数据集包装器和数据管理器
依赖: src.datasets.federated_dataset

代码生成来源: code_generation_guide.md
章节: 阶段3 文件清单
生成日期: 2026-04-16
"""

from .federated_dataset import FederatedDataManager, FederatedDataset

__all__ = ["FederatedDataset", "FederatedDataManager"]
