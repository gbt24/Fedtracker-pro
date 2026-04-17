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
import sys
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.core.config import Config
from src.models import MobileNetV2, ResNet18, ResNet34, VGG11, VGG16


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


def _get_input_channels(dataset_name: str) -> int:
    """根据数据集名称推断输入通道数。"""
    return 1 if dataset_name.lower() == "mnist" else 3


def build_model_from_config(config: Config):
    """根据配置构建分类模型。"""
    model_name = config.model.name.lower()
    num_classes = config.model.num_classes
    input_channels = _get_input_channels(config.data.dataset)

    builders = {
        "resnet18": lambda: ResNet18(
            num_classes=num_classes,
            input_channels=input_channels,
        ),
        "resnet34": lambda: ResNet34(
            num_classes=num_classes,
            input_channels=input_channels,
        ),
        "vgg11": lambda: VGG11(
            num_classes=num_classes,
            input_channels=input_channels,
        ),
        "vgg16": lambda: VGG16(
            num_classes=num_classes,
            input_channels=input_channels,
        ),
        "mobilenetv2": lambda: MobileNetV2(
            num_classes=num_classes,
            input_channels=input_channels,
        ),
        "mobilenet_v2": lambda: MobileNetV2(
            num_classes=num_classes,
            input_channels=input_channels,
        ),
        "mobilenet": lambda: MobileNetV2(
            num_classes=num_classes,
            input_channels=input_channels,
        ),
    }
    if model_name not in builders:
        raise ValueError(
            f"Unsupported classifier model '{config.model.name}'. "
            "Supported: resnet18, resnet34, vgg11, vgg16, mobilenetv2."
        )
    return builders[model_name]()


def resolve_progress_flag(progress: Optional[bool]) -> bool:
    """解析是否启用进度条。"""
    if progress is not None:
        return progress
    return sys.stderr.isatty()


def progress_iter(
    iterable: Iterable,
    *,
    enabled: bool,
    total: int,
    desc: str,
    unit: str,
):
    """根据开关返回带进度条或原始迭代器。"""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:  # pragma: no cover
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)
