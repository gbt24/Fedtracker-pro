"""实验工具函数。

本文件属于 FedTracker-Pro 项目
功能: 提供实验结果保存、目录创建和指标聚合
依赖: json, os, torch

代码生成来源: code_generation_guide.md
章节: 阶段7 实验工具
生成日期: 2026-04-16
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import torch


def _convert_tensors(obj: Any) -> Any:
    """递归将 Tensor 转换为 JSON 可序列化对象。"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {key: _convert_tensors(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_convert_tensors(item) for item in obj]
    return obj


def save_results(
    results: Dict[str, Any], save_dir: str, filename: str = "results.json"
) -> str:
    """保存实验结果到 JSON 文件。

    Args:
        results: 结果字典。
        save_dir: 输出目录。
        filename: 输出文件名。

    Returns:
        结果文件完整路径。
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    serializable = _convert_tensors(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return path


def create_experiment_dir(base_dir: str = "./experiments/results") -> str:
    """创建带时间戳的实验目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def aggregate_client_metrics(
    client_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """聚合客户端指标，返回均值/最小值/最大值。"""
    if not client_metrics:
        return {}

    aggregated: Dict[str, float] = {}
    keys = set().union(*(metrics.keys() for metrics in client_metrics))
    for key in keys:
        values = [metrics[key] for metrics in client_metrics if key in metrics]
        if not values:
            continue
        aggregated[f"{key}_mean"] = float(sum(values) / len(values))
        aggregated[f"{key}_min"] = float(min(values))
        aggregated[f"{key}_max"] = float(max(values))
    return aggregated
