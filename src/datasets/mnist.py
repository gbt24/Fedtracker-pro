"""MNIST 数据集适配器。

本文件属于 FedTracker-Pro 项目
功能: 提供 MNIST 训练与测试集加载函数
依赖: torchvision.datasets, torchvision.transforms

代码生成来源: code_generation_guide.md
章节: 阶段3 数据集与模型
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_mnist_datasets(
    data_dir: str = "./data",
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """加载 MNIST 训练集与测试集。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )
    return train_dataset, test_dataset
