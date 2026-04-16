"""数据集适配器模块单元测试。"""

import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.datasets.cifar import get_cifar10_datasets, get_cifar100_datasets
from src.datasets.mnist import get_mnist_datasets


class _FakeDataset(torch.utils.data.Dataset):
    """最小假数据集，用于拦截下载逻辑。"""

    def __init__(self, *args, **kwargs):
        _ = args
        self.kwargs = kwargs

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        _ = idx
        return torch.randn(3, 32, 32), torch.tensor(1)


class _FakeMNISTDataset(torch.utils.data.Dataset):
    """单通道 MNIST 假数据集。"""

    def __init__(self, *args, **kwargs):
        _ = args
        self.kwargs = kwargs

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        _ = idx
        return torch.randn(1, 28, 28), torch.tensor(1)


class TestCIFARAdapters(unittest.TestCase):
    """测试 CIFAR 数据集适配器。"""

    @patch("src.datasets.cifar.datasets.CIFAR10", new=_FakeDataset)
    def test_get_cifar10_datasets(self) -> None:
        train_ds, test_ds = get_cifar10_datasets(data_dir="./data", download=False)
        self.assertEqual(len(train_ds), 4)
        self.assertEqual(len(test_ds), 4)
        self.assertTrue(train_ds.kwargs["train"])
        self.assertFalse(test_ds.kwargs["train"])
        self.assertFalse(train_ds.kwargs["download"])
        self.assertIn("transform", train_ds.kwargs)

    @patch("src.datasets.cifar.datasets.CIFAR100", new=_FakeDataset)
    def test_get_cifar100_datasets(self) -> None:
        train_ds, test_ds = get_cifar100_datasets(data_dir="./data", download=False)
        self.assertEqual(len(train_ds), 4)
        self.assertEqual(len(test_ds), 4)
        self.assertTrue(train_ds.kwargs["train"])
        self.assertFalse(test_ds.kwargs["train"])
        self.assertFalse(train_ds.kwargs["download"])


class TestMNISTAdapter(unittest.TestCase):
    """测试 MNIST 数据集适配器。"""

    @patch("src.datasets.mnist.datasets.MNIST", new=_FakeMNISTDataset)
    def test_get_mnist_datasets(self) -> None:
        train_ds, test_ds = get_mnist_datasets(data_dir="./data", download=False)
        self.assertEqual(len(train_ds), 4)
        self.assertEqual(len(test_ds), 4)
        sample_x, _ = train_ds[0]
        self.assertEqual(sample_x.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
