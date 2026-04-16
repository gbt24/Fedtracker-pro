"""Aggregation 模块单元测试。"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.aggregation.base_aggregator import BaseAggregator
from src.aggregation.fed_avg import FedAvgAggregator
from src.aggregation.fed_prox import FedProxAggregator


class DummyAggregator(BaseAggregator):
    """用于测试基类工具方法的最小实现。"""

    def aggregate(self, local_states, weights=None, global_state=None):
        _ = weights
        _ = global_state
        return local_states[0]


class TestBaseAggregator(unittest.TestCase):
    """测试聚合器基类工具方法。"""

    def test_compute_weights_uniform(self) -> None:
        agg = DummyAggregator(device="cpu")
        local_states = [{"w": torch.tensor([1.0])} for _ in range(4)]
        weights = agg._compute_weights(local_states)
        self.assertEqual(weights, [0.25, 0.25, 0.25, 0.25])

    def test_compute_weights_by_dataset_size(self) -> None:
        agg = DummyAggregator(device="cpu")
        local_states = [{"w": torch.tensor([1.0])} for _ in range(3)]
        weights = agg._compute_weights(local_states, dataset_sizes=[1, 2, 3])
        self.assertEqual(weights, [1 / 6, 2 / 6, 3 / 6])

    def test_device_fallback_matches_cuda_availability(self) -> None:
        agg = DummyAggregator(device="cuda")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(agg.device, expected)


class TestFedAvgAggregator(unittest.TestCase):
    """测试 FedAvg 聚合行为。"""

    def test_fedavg_weighted_average(self) -> None:
        agg = FedAvgAggregator(device="cpu")
        local_states = [
            {"w": torch.tensor([1.0, 3.0])},
            {"w": torch.tensor([5.0, 7.0])},
        ]
        out = agg.aggregate(local_states, weights=[0.25, 0.75])
        torch.testing.assert_close(out["w"], torch.tensor([4.0, 6.0]))

    def test_fedavg_normalizes_weights(self) -> None:
        agg = FedAvgAggregator(device="cpu")
        local_states = [
            {"w": torch.tensor([1.0])},
            {"w": torch.tensor([5.0])},
        ]
        out = agg.aggregate(local_states, weights=[1.0, 3.0])
        torch.testing.assert_close(out["w"], torch.tensor([4.0]))

    def test_fedavg_rejects_negative_weights(self) -> None:
        agg = FedAvgAggregator(device="cpu")
        local_states = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([2.0])}]
        with self.assertRaises(ValueError):
            agg.aggregate(local_states, weights=[1.0, -1.0])

    def test_fedavg_keeps_non_floating_tensors_from_first_client(self) -> None:
        agg = FedAvgAggregator(device="cpu")
        local_states = [
            {"w": torch.tensor([1.0]), "count": torch.tensor([1], dtype=torch.long)},
            {"w": torch.tensor([3.0]), "count": torch.tensor([9], dtype=torch.long)},
        ]
        out = agg.aggregate(local_states, weights=[0.5, 0.5])
        torch.testing.assert_close(out["w"], torch.tensor([2.0]))
        self.assertEqual(out["count"].dtype, torch.long)
        torch.testing.assert_close(out["count"], torch.tensor([1], dtype=torch.long))

    def test_fedavg_raises_when_no_local_states(self) -> None:
        agg = FedAvgAggregator(device="cpu")
        with self.assertRaises(ValueError):
            agg.aggregate([])

    def test_fedavg_raises_when_weights_length_mismatch(self) -> None:
        agg = FedAvgAggregator(device="cpu")
        local_states = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([2.0])}]
        with self.assertRaises(ValueError):
            agg.aggregate(local_states, weights=[1.0])


class TestFedProxAggregator(unittest.TestCase):
    """测试 FedProx 聚合行为。"""

    def test_fedprox_adds_proximal_term(self) -> None:
        agg = FedProxAggregator(device="cpu", mu=0.5)
        local_states = [
            {"w": torch.tensor([2.0])},
            {"w": torch.tensor([6.0])},
        ]
        global_state = {"w": torch.tensor([10.0])}

        # weighted_sum = 0.5*2 + 0.5*6 = 4
        # prox_term = 0.5 * (0.5*(10-2) + 0.5*(10-6)) = 3
        # result = 7
        out = agg.aggregate(local_states, weights=[0.5, 0.5], global_state=global_state)
        torch.testing.assert_close(out["w"], torch.tensor([7.0]))

    def test_fedprox_raises_when_weights_length_mismatch(self) -> None:
        agg = FedProxAggregator(device="cpu", mu=0.5)
        local_states = [{"w": torch.tensor([2.0])}, {"w": torch.tensor([6.0])}]
        global_state = {"w": torch.tensor([10.0])}
        with self.assertRaises(ValueError):
            agg.aggregate(local_states, weights=[1.0], global_state=global_state)

    def test_fedprox_normalizes_weights(self) -> None:
        agg = FedProxAggregator(device="cpu", mu=0.0)
        local_states = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([5.0])}]
        global_state = {"w": torch.tensor([0.0])}
        out = agg.aggregate(local_states, weights=[1.0, 3.0], global_state=global_state)
        torch.testing.assert_close(out["w"], torch.tensor([4.0]))

    def test_fedprox_rejects_negative_weights(self) -> None:
        agg = FedProxAggregator(device="cpu", mu=0.0)
        local_states = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([2.0])}]
        global_state = {"w": torch.tensor([0.0])}
        with self.assertRaises(ValueError):
            agg.aggregate(local_states, weights=[1.0, -1.0], global_state=global_state)

    def test_fedprox_requires_global_state(self) -> None:
        agg = FedProxAggregator(device="cpu", mu=0.5)
        with self.assertRaises(ValueError):
            agg.aggregate(
                [{"w": torch.tensor([1.0])}], weights=[1.0], global_state=None
            )

    def test_fedprox_keeps_non_floating_tensors_from_first_client(self) -> None:
        agg = FedProxAggregator(device="cpu", mu=0.5)
        local_states = [
            {"w": torch.tensor([2.0]), "count": torch.tensor([5], dtype=torch.long)},
            {"w": torch.tensor([6.0]), "count": torch.tensor([11], dtype=torch.long)},
        ]
        global_state = {
            "w": torch.tensor([10.0]),
            "count": torch.tensor([0], dtype=torch.long),
        }
        out = agg.aggregate(local_states, weights=[0.5, 0.5], global_state=global_state)
        self.assertEqual(out["count"].dtype, torch.long)
        torch.testing.assert_close(out["count"], torch.tensor([5], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
