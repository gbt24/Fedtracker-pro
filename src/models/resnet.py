"""ResNet 模型定义。

本文件属于 FedTracker-Pro 项目
功能: 提供 ResNet 基础块与 ResNet18/ResNet34 工厂函数
依赖: torch, torch.nn

代码生成来源: code_generation_guide.md
章节: 阶段3 数据集与模型
生成日期: 2026-04-16
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet 基础残差块。"""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行残差块前向计算。"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet 主网络。"""

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: list[int],
        num_classes: int = 10,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """构建单个 stage 的残差层。"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for stride_value in strides:
            layers.append(block(self.in_planes, planes, stride_value))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 ResNet 前向传播。"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes: int = 10, input_channels: int = 3) -> ResNet:
    """构建 ResNet-18。"""
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        input_channels=input_channels,
    )


def ResNet34(num_classes: int = 10, input_channels: int = 3) -> ResNet:
    """构建 ResNet-34。"""
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        input_channels=input_channels,
    )
