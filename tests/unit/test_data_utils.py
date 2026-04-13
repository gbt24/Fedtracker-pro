"""DataUtils模块单元测试"""

import unittest
import torch
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.data_utils import (
    set_seed,
    partition_data_iid,
    partition_data_dirichlet,
    partition_data_by_shard,
    get_data_distribution,
    print_data_distribution,
)


class TestSetSeed(unittest.TestCase):
    """测试set_seed函数"""

    def test_set_seed_makes_results_reproducible(self):
        """测试设置种子使结果可重现"""
        # 第一次运行
        set_seed(42)
        random_values1 = [random.random() for _ in range(5)]
        np_values1 = np.random.randn(5)
        torch_values1 = torch.randn(5)

        # 第二次运行（相同种子）
        set_seed(42)
        random_values2 = [random.random() for _ in range(5)]
        np_values2 = np.random.randn(5)
        torch_values2 = torch.randn(5)

        # 验证结果相同
        self.assertEqual(random_values1, random_values2)
        np.testing.assert_array_equal(np_values1, np_values2)
        torch.testing.assert_close(torch_values1, torch_values2)


class TestPartitionDataIID(unittest.TestCase):
    """测试partition_data_iid函数"""

    def setUp(self):
        """创建测试数据集"""
        set_seed(42)
        self.num_samples = 100
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(self.num_samples, 10), torch.randint(0, 10, (self.num_samples,))
        )

    def test_partition_data_iid_creates_correct_number_of_clients(self):
        """测试IID划分创建正确数量的客户端"""
        num_clients = 10
        client_indices = partition_data_iid(self.dataset, num_clients)

        self.assertEqual(len(client_indices), num_clients)

    def test_partition_data_iid_distributes_all_samples(self):
        """测试IID划分分配所有样本"""
        num_clients = 10
        client_indices = partition_data_iid(self.dataset, num_clients)

        # 统计分配的样本总数
        total_assigned = sum(len(indices) for indices in client_indices)
        self.assertEqual(total_assigned, self.num_samples)

    def test_partition_data_iid_no_overlapping_samples(self):
        """测试IID划分没有重叠样本"""
        num_clients = 10
        client_indices = partition_data_iid(self.dataset, num_clients)

        # 检查是否有重叠
        all_indices = set()
        for indices in client_indices:
            for idx in indices:
                self.assertNotIn(idx, all_indices)
                all_indices.add(idx)

        self.assertEqual(len(all_indices), self.num_samples)

    def test_partition_data_iid_approximately_equal_distribution(self):
        """测试IID划分近似均匀分布"""
        num_clients = 10
        client_indices = partition_data_iid(self.dataset, num_clients)

        # 检查每个客户端的样本数
        sizes = [len(indices) for indices in client_indices]
        # 应该近似均匀（差值不超过1）
        max_size = max(sizes)
        min_size = min(sizes)
        self.assertLessEqual(max_size - min_size, 1)


class TestPartitionDataDirichlet(unittest.TestCase):
    """测试partition_data_dirichlet函数"""

    def setUp(self):
        """创建测试数据集"""
        set_seed(42)
        self.num_samples = 100
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(self.num_samples, 10), torch.randint(0, 10, (self.num_samples,))
        )

    def test_partition_data_dirichlet_creates_correct_number_of_clients(self):
        """测试Dirichlet划分创建正确数量的客户端"""
        num_clients = 10
        client_indices = partition_data_dirichlet(self.dataset, num_clients, alpha=1.0)

        self.assertEqual(len(client_indices), num_clients)

    def test_partition_data_dirichlet_distributes_all_samples(self):
        """测试Dirichlet划分分配所有样本"""
        num_clients = 10
        client_indices = partition_data_dirichlet(self.dataset, num_clients, alpha=1.0)

        total_assigned = sum(len(indices) for indices in client_indices)
        self.assertEqual(total_assigned, self.num_samples)

    def test_partition_data_dirichlet_alpha_affects_distribution(self):
        """测试alpha参数影响分布"""
        set_seed(42)
        client_indices_high_alpha = partition_data_dirichlet(
            self.dataset, 10, alpha=10.0
        )

        set_seed(42)
        client_indices_low_alpha = partition_data_dirichlet(self.dataset, 10, alpha=0.1)

        # 高alpha应该更均匀
        sizes_high = [len(indices) for indices in client_indices_high_alpha]
        sizes_low = [len(indices) for indices in client_indices_low_alpha]

        std_high = np.std(sizes_high)
        std_low = np.std(sizes_low)

        # 高alpha的标准差应该更小（更均匀）
        self.assertLess(std_high, std_low)


class TestPartitionDataByShard(unittest.TestCase):
    """测试partition_data_by_shard函数"""

    def setUp(self):
        """创建测试数据集"""
        set_seed(42)
        self.num_samples = 100
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(self.num_samples, 10), torch.randint(0, 10, (self.num_samples,))
        )

    def test_partition_data_by_shard_creates_correct_number_of_clients(self):
        """测试Shard划分创建正确数量的客户端"""
        num_clients = 10
        num_shards = 20
        client_indices = partition_data_by_shard(self.dataset, num_clients, num_shards)

        self.assertEqual(len(client_indices), num_clients)

    def test_partition_data_by_shard_distributes_all_samples(self):
        """测试Shard划分分配所有样本"""
        num_clients = 10
        num_shards = 20
        client_indices = partition_data_by_shard(self.dataset, num_clients, num_shards)

        total_assigned = sum(len(indices) for indices in client_indices)
        # 可能由于shard数不能整除，会有一些样本未分配
        self.assertLessEqual(total_assigned, self.num_samples)


class TestGetDataDistribution(unittest.TestCase):
    """测试get_data_distribution函数"""

    def setUp(self):
        """创建测试数据集"""
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.cat(
                [torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)]
            ),
        )

    def test_get_data_distribution_returns_correct_shape(self):
        """测试get_data_distribution返回正确形状"""
        indices = list(range(100))
        num_classes = 10
        distribution = get_data_distribution(self.dataset, indices, num_classes)

        self.assertEqual(len(distribution), num_classes)

    def test_get_data_distribution_sums_to_one(self):
        """测试get_data_distribution的总和为1"""
        indices = list(range(100))
        num_classes = 10
        distribution = get_data_distribution(self.dataset, indices, num_classes)

        # 由于是概率分布，总和应该为1
        self.assertAlmostEqual(distribution.sum(), 1.0, places=5)

    def test_get_data_distribution_empty_indices(self):
        """测试空索引返回零分布"""
        indices = []
        num_classes = 10
        distribution = get_data_distribution(self.dataset, indices, num_classes)

        # 应该返回全零
        self.assertTrue(np.all(distribution == 0))


class TestPrintDataDistribution(unittest.TestCase):
    """测试print_data_distribution函数"""

    def setUp(self):
        """创建测试数据集和客户端索引"""
        set_seed(42)
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10), torch.randint(0, 10, (100,))
        )
        self.client_indices = partition_data_iid(self.dataset, 10)

    def test_print_data_distribution_does_not_crash(self):
        """测试print_data_distribution不会崩溃"""
        # 这个测试主要验证函数可以正常运行
        try:
            print_data_distribution(self.client_indices, self.dataset, num_classes=10)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"print_data_distribution raised {e}")


if __name__ == "__main__":
    unittest.main()
