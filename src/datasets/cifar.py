"""CIFAR 数据集适配器。

本文件属于 FedTracker-Pro 项目
功能: 提供 CIFAR-10/CIFAR-100 训练与测试集加载函数
依赖: torchvision.datasets, torchvision.transforms

代码生成来源: code_generation_guide.md
章节: 阶段3 数据集与模型
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _cifar_transform() -> transforms.Compose:
    """返回 CIFAR 通用预处理。"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def get_cifar10_datasets(
    data_dir: str = "./data",
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """加载 CIFAR-10 训练集与测试集。"""
    transform = _cifar_transform()
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )
    return train_dataset, test_dataset


def get_cifar100_datasets(
    data_dir: str = "./data",
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """加载 CIFAR-100 训练集与测试集。"""
    transform = _cifar_transform()
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )
    return train_dataset, test_dataset
