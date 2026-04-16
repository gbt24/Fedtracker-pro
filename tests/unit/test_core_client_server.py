"""Core 客户端与服务器模块单元测试。"""

import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.base_client import StandardClient
from src.core.base_server import BaseServer
from src.aggregation.fed_avg import FedAvgAggregator


class TinyNet(nn.Module):
    """测试用最小模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class EchoAggregator:
    """用于测试服务器聚合流程的假聚合器。"""

    def aggregate(self, local_states, global_state, weights=None):
        _ = global_state
        _ = weights
        return local_states[0]


def make_loader() -> torch.utils.data.DataLoader:
    """构造小型分类数据加载器。"""
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)


class TestStandardClient(unittest.TestCase):
    """测试标准客户端训练和测试流程。"""

    def test_local_train_and_local_test(self) -> None:
        model = TinyNet()
        loader = make_loader()
        client = StandardClient(
            client_id=1,
            model=model,
            train_loader=loader,
            test_loader=loader,
            device="cpu",
            local_epochs=1,
            local_lr=0.1,
            optimizer_name="sgd",
        )

        state = client.local_train()
        self.assertIn("linear.weight", state)
        self.assertGreaterEqual(len(client.training_history), 1)

        metrics = client.local_test()
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)

    def test_unknown_optimizer_raises(self) -> None:
        client = StandardClient(
            client_id=2,
            model=TinyNet(),
            train_loader=make_loader(),
            device="cpu",
            optimizer_name="unknown",
        )
        with self.assertRaises(ValueError):
            client.local_train()


class TestBaseServer(unittest.TestCase):
    """测试服务器聚合、评估与检查点流程。"""

    def test_aggregate_evaluate_and_checkpoint(self) -> None:
        model = TinyNet()
        server = BaseServer(
            model=model,
            aggregator=EchoAggregator(),
            device="cpu",
        )

        local_state = {
            "linear.weight": torch.full_like(model.linear.weight.data, 2.0),
            "linear.bias": torch.full_like(model.linear.bias.data, 1.0),
        }
        server.aggregate([local_state])

        self.assertEqual(server.round_num, 1)
        state = server.get_global_state()
        torch.testing.assert_close(state["linear.bias"], torch.tensor([1.0, 1.0]))

        metrics = server.evaluate(make_loader())
        self.assertEqual(metrics["round"], 1)
        self.assertIn("accuracy", metrics)
        self.assertIn("loss", metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = os.path.join(tmp_dir, "server.pt")
            server.save_checkpoint(ckpt)

            restored = BaseServer(
                model=TinyNet(),
                aggregator=EchoAggregator(),
                device="cpu",
            )
            restored.load_checkpoint(ckpt)
            self.assertEqual(restored.round_num, 1)
            self.assertEqual(len(restored.history), 1)

    def test_server_works_with_fedavg_aggregator(self) -> None:
        model = TinyNet()
        server = BaseServer(
            model=model,
            aggregator=FedAvgAggregator(device="cpu"),
            device="cpu",
        )
        local_state_1 = {
            "linear.weight": torch.full_like(model.linear.weight.data, 1.0),
            "linear.bias": torch.full_like(model.linear.bias.data, 1.0),
        }
        local_state_2 = {
            "linear.weight": torch.full_like(model.linear.weight.data, 3.0),
            "linear.bias": torch.full_like(model.linear.bias.data, 3.0),
        }

        server.aggregate([local_state_1, local_state_2], weights=[0.5, 0.5])
        state = server.get_global_state()
        torch.testing.assert_close(state["linear.bias"], torch.tensor([2.0, 2.0]))

    def test_load_checkpoint_rejects_invalid_structure(self) -> None:
        server = BaseServer(
            model=TinyNet(),
            aggregator=EchoAggregator(),
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_ckpt = os.path.join(tmp_dir, "bad.pt")
            torch.save({"bad": "format"}, bad_ckpt)
            with self.assertRaises(ValueError):
                server.load_checkpoint(bad_ckpt)


if __name__ == "__main__":
    unittest.main()
