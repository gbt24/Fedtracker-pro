"""VGG 模型定义。

本文件属于 FedTracker-Pro 项目
功能: 提供 VGG11/VGG16 的 CIFAR 版本实现
依赖: torch, torch.nn

代码生成来源: code_generation_guide.md
章节: 阶段3 数据集与模型
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn


CfgItem = Union[int, str]


_VGG_CFGS: dict[str, List[CfgItem]] = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    """VGG 主网络。"""

    def __init__(
        self,
        vgg_name: str,
        num_classes: int = 10,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if vgg_name not in _VGG_CFGS:
            raise ValueError(f"Unsupported VGG variant: {vgg_name}")

        self.features = self._make_layers(_VGG_CFGS[vgg_name], input_channels)
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg: List[CfgItem], in_channels: int) -> nn.Sequential:
        """按配置构建卷积特征提取层。"""
        layers: list[nn.Module] = []
        for layer_cfg in cfg:
            if layer_cfg == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(layer_cfg)
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向传播。"""
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def VGG11(num_classes: int = 10, input_channels: int = 3) -> VGG:
    """构建 VGG11。"""
    return VGG("VGG11", num_classes=num_classes, input_channels=input_channels)


def VGG16(num_classes: int = 10, input_channels: int = 3) -> VGG:
    """构建 VGG16。"""
    return VGG("VGG16", num_classes=num_classes, input_channels=input_channels)
