"""轻量级 Diffusion U-Net 模型。

本文件属于 FedTracker-Pro 项目
功能: 提供最小可用的扩散去噪网络骨干
依赖: torch, torch.nn

代码生成来源: implementation_plan.md
章节: 阶段3 模型定义
生成日期: 2026-04-16
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间步编码。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be greater than 0")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            raise ValueError("t must be a 1D tensor")
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(-torch.arange(half_dim, device=t.device) * emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        sinusoidal = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        if self.dim % 2 == 1:
            sinusoidal = torch.nn.functional.pad(sinusoidal, (0, 1))
        return sinusoidal


class DiffusionUNet(nn.Module):
    """轻量 U-Net 去噪器。"""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if base_channels <= 0:
            raise ValueError("base_channels must be greater than 0")

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
        )
        self.time_proj = nn.Linear(base_channels, base_channels * 2)

        self.input_proj = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        self.down = nn.Sequential(
            nn.Conv2d(
                base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.output_proj = nn.Conv2d(
            base_channels, in_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("x must be a 4D tensor [B, C, H, W]")
        if t.shape[0] != x.shape[0]:
            raise ValueError("t batch size must match x batch size")
        if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
            raise ValueError(
                f"Spatial dims must be even, got {x.shape[2]}x{x.shape[3]}. "
                "Pad inputs to even dimensions before passing to DiffusionUNet."
            )

        time_feat = self.time_proj(self.time_embed(t)).unsqueeze(-1).unsqueeze(-1)
        h0 = self.input_proj(x)
        h1 = self.down(h0)
        h1 = h1 + time_feat
        h2 = self.mid(h1)
        h3 = self.up(h2)
        h = h3 + h0
        return self.output_proj(h)
