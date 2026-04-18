"""持续学习后门水印实现。

本文件属于 FedTracker-Pro 项目
功能: 提供基于触发集与简化 GEM 记忆机制的水印嵌入与验证
依赖: torch, torch.nn, src.defense.watermark.base_watermark

代码生成来源: code_generation_guide.md
章节: 阶段4 水印系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_watermark import BaseWatermark


class ContinualLearningWatermark(BaseWatermark):
    """基于持续学习的后门水印。"""

    def __init__(
        self,
        trigger_size: int = 100,
        target_label: int = 0,
        device: str = "cuda",
        memory_size: int = 100,
        gem_margin: float = 0.5,
    ) -> None:
        super().__init__(trigger_size, target_label, device)
        self.memory_size = memory_size
        self.gem_margin = gem_margin
        self.episodic_memory: list[Tuple[torch.Tensor, torch.Tensor]] = []

    def reset_memory(self) -> None:
        """清空 episodic memory，避免跨客户端样本泄漏。"""
        self.episodic_memory = []

    def generate_trigger_set(
        self,
        data_loader: Optional[torch.utils.data.DataLoader] = None,
        pattern_type: str = "checkerboard",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成触发集。"""
        if data_loader is not None:
            try:
                sample = next(iter(data_loader))[0]
            except StopIteration as exc:
                raise ValueError("data_loader is empty") from exc
            data_shape = tuple(sample.shape[1:])
        else:
            data_shape = (3, 32, 32)

        triggers = []
        for _ in range(self.trigger_size):
            if pattern_type == "checkerboard":
                trigger = self._create_checkerboard(data_shape)
            elif pattern_type == "random":
                trigger = torch.randn(data_shape)
            elif pattern_type == "waffle":
                trigger = self._create_waffle_pattern(data_shape)
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
            triggers.append(trigger)

        trigger_tensor = torch.stack(triggers)
        target_tensor = torch.full(
            (self.trigger_size,),
            self.target_label,
            dtype=torch.long,
        )
        self.trigger_set = (trigger_tensor, target_tensor)
        return self.trigger_set

    def _create_checkerboard(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """创建棋盘格触发模式。"""
        c, h, w = shape
        pattern = torch.zeros((c, h, w))
        for i in range(h):
            for j in range(w):
                if (i + j) % 2 == 0:
                    pattern[:, i, j] = 1.0
        return pattern

    def _create_waffle_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """创建 waffle 触发模式。"""
        c, h, w = shape
        pattern = torch.randn((c, h, w)) * 0.1
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                pattern[:, i : min(i + 2, h), j : min(j + 2, w)] = 1.0
        return pattern

    def embed(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 0.001,
        main_task_weight: float = 1.0,
        watermark_weight: float = 1.0,
    ) -> nn.Module:
        """嵌入水印。"""
        if self.trigger_set is None:
            self.generate_trigger_set(train_loader)

        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        trigger_data, trigger_targets = self.trigger_set
        trigger_data = trigger_data.to(self.device)
        trigger_targets = trigger_targets.to(self.device)

        for _ in range(epochs):
            for data, target in train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output_main = model(data)
                loss_main = F.cross_entropy(output_main, target)

                output_wm = model(trigger_data)
                loss_wm = F.cross_entropy(output_wm, trigger_targets)

                replay_loss = torch.tensor(0.0, device=self.device)
                if self.episodic_memory:
                    mem_data = torch.cat(
                        [item[0] for item in self.episodic_memory], dim=0
                    )
                    mem_target = torch.cat(
                        [item[1] for item in self.episodic_memory], dim=0
                    )
                    mem_data = mem_data.to(self.device)
                    mem_target = mem_target.to(self.device)
                    replay_output = model(mem_data)
                    replay_loss = F.cross_entropy(replay_output, mem_target)

                loss = (
                    main_task_weight * loss_main
                    + watermark_weight * loss_wm
                    + self.gem_margin * replay_loss
                )
                loss.backward()
                optimizer.step()

                self._update_memory(data.detach(), target.detach())

        return model

    def _update_memory(self, data: torch.Tensor, target: torch.Tensor) -> None:
        """维护简化 episodic memory。"""
        if len(self.episodic_memory) >= self.memory_size:
            self.episodic_memory.pop(0)
        self.episodic_memory.append((data[:1].clone(), target[:1].clone()))

    def verify(self, model: nn.Module) -> float:
        """验证触发集分类准确率。"""
        if self.trigger_set is None:
            raise ValueError("Trigger set not generated")

        model = model.to(self.device)
        model.eval()

        trigger_data, trigger_targets = self.trigger_set
        trigger_data = trigger_data.to(self.device)
        trigger_targets = trigger_targets.to(self.device)

        with torch.no_grad():
            outputs = model(trigger_data)
            predicted = outputs.argmax(dim=1)
            accuracy = predicted.eq(trigger_targets).float().mean().item()

        return float(accuracy)
