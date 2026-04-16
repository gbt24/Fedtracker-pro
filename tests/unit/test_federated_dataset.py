"""Federated dataset 模块单元测试。"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.datasets.federated_dataset import FederatedDataManager, FederatedDataset


class DummyDataManager(FederatedDataManager):
    """测试用数据管理器，避免下载真实数据集。"""

    def _load_dataset(self):
        x_train = torch.randn(100, 3, 32, 32)
        y_train = torch.randint(0, 10, (100,))
        x_test = torch.randn(20, 3, 32, 32)
        y_test = torch.randint(0, 10, (20,))
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        return train_dataset, test_dataset


class SmallDummyDataManager(FederatedDataManager):
    """小样本数据管理器，用于边界条件测试。"""

    def _load_dataset(self):
        x_train = torch.randn(7, 3, 32, 32)
        y_train = torch.randint(0, 10, (7,))
        x_test = torch.randn(4, 3, 32, 32)
        y_test = torch.randint(0, 10, (4,))
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        return train_dataset, test_dataset


class TestFederatedDataset(unittest.TestCase):
    """测试 FederatedDataset 包装器行为。"""

    def test_dataset_length_and_index_mapping(self) -> None:
        dataset = torch.utils.data.TensorDataset(
            torch.arange(10).float().unsqueeze(1),
            torch.arange(10).long(),
        )
        wrapped = FederatedDataset(dataset, indices=[2, 5, 8])
        self.assertEqual(len(wrapped), 3)
        feature, label = wrapped[1]
        torch.testing.assert_close(feature, torch.tensor([5.0]))
        self.assertEqual(int(label), 5)


class TestFederatedDataManager(unittest.TestCase):
    """测试数据管理器分区与 dataloader。"""

    def test_iid_partition_and_client_loader(self) -> None:
        manager = DummyDataManager(dataset_name="dummy", num_clients=5, iid=True)
        self.assertEqual(len(manager.client_indices), 5)
        total = sum(len(indices) for indices in manager.client_indices)
        self.assertEqual(total, len(manager.train_dataset))

        loader = manager.get_client_loader(client_id=0, batch_size=8, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        self.assertEqual(batch_x.dim(), 4)
        self.assertEqual(batch_y.dim(), 1)

    def test_dirichlet_partition_returns_all_clients(self) -> None:
        manager = DummyDataManager(
            dataset_name="dummy",
            num_clients=4,
            iid=False,
            alpha=0.5,
        )
        self.assertEqual(len(manager.client_indices), 4)
        test_loader = manager.get_test_loader(batch_size=16)
        test_x, test_y = next(iter(test_loader))
        self.assertEqual(test_x.shape[0], 16)
        self.assertEqual(test_y.shape[0], 16)

    def test_shard_partition_path(self) -> None:
        manager = DummyDataManager(
            dataset_name="dummy",
            num_clients=5,
            iid=False,
            alpha=0.0,
            num_shards=20,
        )
        self.assertEqual(len(manager.client_indices), 5)

    def test_get_client_loader_raises_for_invalid_client(self) -> None:
        manager = DummyDataManager(dataset_name="dummy", num_clients=3, iid=True)
        with self.assertRaises(ValueError):
            manager.get_client_loader(client_id=3)


class TestFederatedDataManagerValidation(unittest.TestCase):
    """测试数据管理器异常路径。"""

    def test_unsupported_dataset_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            FederatedDataManager(dataset_name="unsupported_dataset", num_clients=2)

    def test_iid_partition_covers_all_samples_when_not_divisible(self) -> None:
        manager = SmallDummyDataManager(dataset_name="dummy", num_clients=4, iid=True)
        flat_indices = [idx for indices in manager.client_indices for idx in indices]
        self.assertEqual(len(flat_indices), len(manager.train_dataset))
        self.assertEqual(len(set(flat_indices)), len(manager.train_dataset))

    def test_iid_partition_raises_when_clients_exceed_samples(self) -> None:
        with self.assertRaises(ValueError):
            SmallDummyDataManager(dataset_name="dummy", num_clients=8, iid=True)


if __name__ == "__main__":
    unittest.main()
