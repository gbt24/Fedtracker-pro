"""数据处理工具"""

import torch
import numpy as np
from typing import List
import random


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def partition_data_iid(
    dataset: torch.utils.data.Dataset, num_clients: int
) -> List[List[int]]:
    """
    IID数据划分

    均匀随机分配数据到各客户端
    """
    num_items = len(dataset)
    items_per_client = num_items // num_clients
    client_indices = [[] for _ in range(num_clients)]

    # 随机打乱
    all_indices = list(range(num_items))
    random.shuffle(all_indices)

    # 均匀分配
    for i, idx in enumerate(all_indices):
        client_id = i // items_per_client
        if client_id < num_clients:
            client_indices[client_id].append(idx)

    return client_indices


def partition_data_dirichlet(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    alpha: float = 0.5,
    num_classes: int = 10,
) -> List[List[int]]:
    """
    非IID数据划分 (Dirichlet分布)

    Args:
        dataset: 数据集
        num_clients: 客户端数量
        alpha: Dirichlet浓度参数，越小越异构
        num_classes: 类别数
    """
    # 按类别组织数据索引
    class_indices = [[] for _ in range(num_classes)]

    for idx, (_, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(idx)

    # 为每个类别分配数据到客户端
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # 生成Dirichlet分布
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # 分配数据（使用floor确保整数）
        counts = (proportions * len(class_indices[c])).astype(int)

        # 处理剩余的样本
        remaining = len(class_indices[c]) - np.sum(counts)
        for k in np.argsort(-proportions)[:remaining]:
            counts[k] += 1

        # 分配数据
        start_idx = 0
        for k in range(num_clients):
            end_idx = start_idx + counts[k]
            client_indices[k].extend(class_indices[c][start_idx:end_idx])
            start_idx = end_idx

    return client_indices


def partition_data_by_shard(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    num_shards: int = 200,
    num_classes: int = 10,
) -> List[List[int]]:
    """
    基于Shard的非IID划分

    将数据分成num_shards个shard，每个客户端分配2个shard
    """
    # 按标签排序
    sorted_indices = sorted(
        range(len(dataset)),
        key=lambda idx: dataset[idx][1]
        if isinstance(dataset[idx][1], int)
        else dataset[idx][1].item(),
    )

    # 创建shard
    shard_size = len(dataset) // num_shards
    shards = [
        sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)
    ]

    # 随机分配shard给客户端
    random.shuffle(shards)

    client_indices = [[] for _ in range(num_clients)]
    shards_per_client = num_shards // num_clients

    for i in range(num_clients):
        for j in range(shards_per_client):
            shard_idx = i * shards_per_client + j
            if shard_idx < len(shards):
                client_indices[i].extend(shards[shard_idx])

    return client_indices


def get_data_distribution(
    dataset: torch.utils.data.Dataset, indices: List[int], num_classes: int = 10
) -> np.ndarray:
    """获取数据分布"""
    distribution = np.zeros(num_classes)

    for idx in indices:
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        distribution[label] += 1

    return distribution / len(indices) if len(indices) > 0 else distribution


def print_data_distribution(
    client_indices: List[List[int]],
    dataset: torch.utils.data.Dataset,
    num_classes: int = 10,
):
    """打印各客户端的数据分布"""
    print("Data Distribution:")
    for i, indices in enumerate(client_indices):
        dist = get_data_distribution(dataset, indices, num_classes)
        print(f"Client {i}: {dist}")
