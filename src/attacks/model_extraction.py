"""模型提取攻击实现。"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_attack import BaseAttack


class ModelExtractionAttack(BaseAttack):
    """通过查询蒸馏训练替代模型。"""

    def attack(
        self,
        model: nn.Module | None = None,
        query_loader: torch.utils.data.DataLoader | None = None,
        surrogate_model: nn.Module | None = None,
        epochs: int = 3,
        lr: float = 0.001,
        **kwargs: object,
    ) -> nn.Module:
        if model is None:
            legacy_model = kwargs.pop("victim_model", None)
            if isinstance(legacy_model, nn.Module):
                model = legacy_model
        if model is None:
            raise ValueError("model must be provided")
        if query_loader is None:
            raise ValueError("query_loader must be provided")
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"unexpected keyword arguments: {unknown}")

        victim_model = model.to(self.device)
        victim_model.eval()

        if surrogate_model is None:
            surrogate_model = copy.deepcopy(victim_model)
        surrogate_model = surrogate_model.to(self.device)
        surrogate_model.train()

        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=lr)

        for _ in range(epochs):
            for data, _ in query_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    teacher_logits = victim_model(data)
                    teacher_labels = teacher_logits.argmax(dim=1)

                optimizer.zero_grad()
                student_logits = surrogate_model(data)
                loss = F.cross_entropy(student_logits, teacher_labels)
                loss.backward()
                optimizer.step()

        return surrogate_model

    def get_attack_name(self) -> str:
        return "model_extraction"
