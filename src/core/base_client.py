"""联邦学习客户端基类。

本文件属于 FedTracker-Pro 项目
功能: 定义客户端训练、测试与保护嵌入接口
依赖: torch, torch.nn, abc

代码生成来源: code_generation_guide.md
章节: 阶段7 客户端基类
生成日期: 2026-04-16
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import copy


class BaseClient(ABC):
    """联邦学习客户端基类。"""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda",
        local_epochs: int = 5,
        local_lr: float = 0.01,
        optimizer_name: str = "sgd",
        **optimizer_kwargs,
    ) -> None:
        self.client_id = client_id
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.local_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.training_history: list[Dict[str, float]] = []

    def get_model_state(self, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
        """获取模型状态。"""
        if to_cpu:
            return {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """设置模型状态。"""
        self.model.load_state_dict(state_dict)

    def local_train(
        self,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
        return_cpu_state: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """执行本地训练。"""
        if global_state is not None:
            self.set_model_state(global_state)

        self.local_model_state = copy.deepcopy(self.get_model_state(to_cpu=True))
        self._train_epoch()
        return self.get_model_state(to_cpu=return_cpu_state)

    def _train_epoch(self) -> None:
        """执行本地训练循环。"""
        self.model.train()
        optimizer = self._get_optimizer()
        criterion = nn.CrossEntropyLoss()
        if len(self.train_loader) == 0:
            raise ValueError("train_loader is empty")

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for data, target in self.train_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                _, predicted = output.max(1)
                total += int(target.size(0))
                correct += int(predicted.eq(target).sum().item())

            accuracy = 100.0 * correct / total if total else 0.0
            avg_loss = epoch_loss / len(self.train_loader)
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(avg_loss),
                    "accuracy": float(accuracy),
                }
            )

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """根据配置返回优化器。"""
        if self.optimizer_name.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.local_lr,
                momentum=self.optimizer_kwargs.get("momentum", 0.9),
                weight_decay=self.optimizer_kwargs.get("weight_decay", 0.0001),
            )
        if self.optimizer_name.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.local_lr,
                weight_decay=self.optimizer_kwargs.get("weight_decay", 0.0001),
            )
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def local_test(self) -> Dict[str, float]:
        """执行本地测试。"""
        if self.test_loader is None:
            return {}
        if len(self.test_loader) == 0:
            raise ValueError("test_loader is empty")

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(data)
                test_loss += float(criterion(output, target).item())
                _, predicted = output.max(1)
                total += int(target.size(0))
                correct += int(predicted.eq(target).sum().item())

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total if total else 0.0
        return {"loss": float(avg_loss), "accuracy": float(accuracy)}

    @abstractmethod
    def embed_protection(self, **kwargs) -> None:
        """嵌入保护措施。"""


class StandardClient(BaseClient):
    """标准联邦学习客户端。"""

    def embed_protection(self, **kwargs) -> None:
        """默认不嵌入保护。"""
