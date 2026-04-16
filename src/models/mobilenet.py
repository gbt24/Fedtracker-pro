"""MobileNetV2 模型定义。

本文件属于 FedTracker-Pro 项目
功能: 提供轻量化 MobileNetV2 分类网络
依赖: torch, torch.nn

代码生成来源: code_generation_guide.md
章节: 阶段3 数据集与模型
生成日期: 2026-04-16
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行倒残差块前向传播。"""
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class _MobileNetV2(nn.Module):
    """MobileNetV2 主体。"""

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__()
        self.last_channel = 1280

        features: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
            )
        ]

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        in_channels = 32
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, stride, t))
                in_channels = c

        features.append(
            nn.Sequential(
                nn.Conv2d(in_channels, self.last_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6(inplace=True),
            )
        )

        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.last_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向传播。"""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def MobileNetV2(num_classes: int = 10, input_channels: int = 3) -> _MobileNetV2:
    """构建 MobileNetV2。"""
    return _MobileNetV2(num_classes=num_classes, input_channels=input_channels)
