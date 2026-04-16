"""联邦学习服务器基类。

本文件属于 FedTracker-Pro 项目
功能: 定义服务器聚合、评估、检查点保存接口
依赖: torch, torch.nn

代码生成来源: code_generation_guide.md
章节: 阶段2/7 服务器基类
生成日期: 2026-04-16
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BaseServer:
    """联邦学习服务器基类。"""

    def __init__(
        self,
        model: nn.Module,
        aggregator,
        device: str = "cuda",
        global_lr: float = 1.0,
    ) -> None:
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.global_model = model.to(self.device)
        self.aggregator = aggregator
        self.global_lr = global_lr
        self.round_num = 0
        self.history: List[Dict[str, float]] = []
        self.client_states: Dict[int, Dict[str, torch.Tensor]] = {}

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数。"""
        return {
            key: value.detach().cpu().clone()
            for key, value in self.global_model.state_dict().items()
        }

    def set_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """设置全局模型参数。"""
        self.global_model.load_state_dict(state_dict)

    def aggregate(
        self,
        local_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> None:
        """执行全局聚合。"""
        if not local_states:
            raise ValueError("local_states cannot be empty")
        current_state = self.get_global_state()
        aggregated_state = self.aggregator.aggregate(
            local_states=local_states,
            weights=weights,
            global_state=current_state,
        )
        self.set_global_state(aggregated_state)
        self.round_num += 1

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估全局模型。"""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        if len(test_loader) == 0:
            raise ValueError("test_loader is empty")
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.global_model(data)
                test_loss += float(criterion(output, target).item())
                _, predicted = output.max(1)
                total += int(target.size(0))
                correct += int(predicted.eq(target).sum().item())

        avg_loss = test_loss / len(test_loader)
        accuracy = 100.0 * correct / total if total else 0.0

        metrics = {
            "round": self.round_num,
            "accuracy": float(accuracy),
            "loss": float(avg_loss),
        }
        self.history.append(metrics)
        return metrics

    def save_checkpoint(self, filepath: str) -> None:
        """保存检查点。"""
        checkpoint = {
            "round": self.round_num,
            "model_state": self.get_global_state(),
            "history": self.history,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """加载检查点。"""
        try:
            checkpoint = torch.load(
                filepath,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)

        required_keys = {"round", "model_state", "history"}
        if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
            raise ValueError("Invalid checkpoint format")

        self.round_num = int(checkpoint["round"])
        self.set_global_state(checkpoint["model_state"])
        self.history = checkpoint["history"]
