"""联邦数据集管理模块。

本文件属于 FedTracker-Pro 项目
功能: 提供联邦数据集包装器与客户端数据分区管理
依赖: torch, torchvision, src.utils.data_utils

代码生成来源: code_generation_guide.md
章节: 阶段3 数据集与模型
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..utils.data_utils import (
    partition_data_by_shard,
    partition_data_dirichlet,
)


class FederatedDataset(Dataset):
    """联邦数据集包装器。

    Args:
        dataset: 原始数据集对象。
        indices: 当前客户端拥有的样本索引列表。
    """

    def __init__(self, dataset: Dataset, indices: Optional[List[int]] = None) -> None:
        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))

    def __len__(self) -> int:
        """返回当前子数据集大小。"""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """按本地索引获取样本。"""
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]


class FederatedDataManager:
    """联邦学习数据管理器。"""

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data",
        num_clients: int = 10,
        iid: bool = True,
        alpha: float = 0.5,
        num_shards: int = 200,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.iid = iid
        self.alpha = alpha
        self.num_shards = num_shards

        self.train_dataset, self.test_dataset = self._load_dataset()
        self.client_indices = self._partition_data()

    def _load_dataset(self) -> Tuple[Dataset, Dataset]:
        """根据名称加载数据集。

        Returns:
            训练集和测试集二元组。

        Raises:
            ValueError: 数据集名称不受支持时抛出。
        """
        if self.dataset_name == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            train = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=transform,
            )
            test = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform,
            )
            return train, test

        if self.dataset_name == "mnist":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            train = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=transform,
            )
            test = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform,
            )
            return train, test

        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _partition_data(self) -> List[List[int]]:
        """划分训练数据到客户端。"""
        if self.iid:
            return self._partition_iid()
        if self.alpha > 0:
            return partition_data_dirichlet(
                self.train_dataset,
                self.num_clients,
                self.alpha,
            )
        return partition_data_by_shard(
            self.train_dataset,
            self.num_clients,
            self.num_shards,
        )

    def _partition_iid(self) -> List[List[int]]:
        """执行鲁棒 IID 划分，保证样本不丢失。"""
        dataset_size = len(self.train_dataset)
        if self.num_clients <= 0:
            raise ValueError("num_clients must be greater than 0")
        if self.num_clients > dataset_size:
            raise ValueError("num_clients cannot exceed dataset size")

        shuffled = torch.randperm(dataset_size).tolist()
        split_arrays = torch.tensor(shuffled).split(
            [
                dataset_size // self.num_clients
                + (1 if i < dataset_size % self.num_clients else 0)
                for i in range(self.num_clients)
            ]
        )
        return [split.tolist() for split in split_arrays]

    def get_client_loader(
        self,
        client_id: int,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """获取指定客户端的数据加载器。"""
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError(f"Invalid client_id: {client_id}")

        dataset = FederatedDataset(self.train_dataset, self.client_indices[client_id])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(self, batch_size: int = 100) -> DataLoader:
        """获取全局测试集加载器。"""
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
