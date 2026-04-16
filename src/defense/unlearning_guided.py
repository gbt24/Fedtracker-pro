"""遗忘增强定位模块。

本文件属于 FedTracker-Pro 项目
功能: 识别稳定参数并对指纹进行频谱增强重定位
依赖: torch, numpy, pywt

代码生成来源: code_generation_guide.md
章节: 阶段5 遗忘增强定位
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnlearningGuidedRelocation:
    """利用模拟遗忘识别稳定参数并重定位指纹。"""

    def __init__(
        self,
        simulation_steps: int = 100,
        stability_threshold: float = 0.8,
        low_freq_ratio: float = 0.8,
        device: str = "cuda",
    ) -> None:
        self.simulation_steps = simulation_steps
        self.stability_threshold = stability_threshold
        self.low_freq_ratio = low_freq_ratio
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )

    def identify_stable_parameters(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        sample_ratio: float = 0.1,
    ) -> Dict[str, float]:
        """通过模拟遗忘评估参数稳定性。"""
        model = model.to(self.device)
        was_training = model.training
        model.eval()
        original = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        max_batches = max(1, int(len(data_loader) * sample_ratio))

        for _ in range(self.simulation_steps):
            batch_count = 0
            for data, target in data_loader:
                if batch_count >= max_batches:
                    break
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                (-loss).backward()
                optimizer.step()
                batch_count += 1

        scores: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if name in original:
                delta = torch.abs(param.detach() - original[name])
                avg_delta = torch.mean(delta).item()
                scores[name] = 1.0 / (1.0 + avg_delta)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original:
                    param.copy_(original[name])
        if was_training:
            model.train()

        return scores

    def select_stable_parameters(
        self,
        stability_scores: Dict[str, float],
        top_k: Optional[int] = None,
    ) -> List[str]:
        """选择稳定参数名列表。"""
        sorted_items = sorted(
            stability_scores.items(), key=lambda item: item[1], reverse=True
        )
        if top_k is not None:
            sorted_items = sorted_items[:top_k]
        return [
            name for name, score in sorted_items if score >= self.stability_threshold
        ]

    def spectral_enhancement(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """保留低频小波系数以增强稳健性。"""
        fp_np = fingerprint.detach().cpu().numpy()
        coeffs = pywt.wavedec(
            fp_np,
            "db1",
            level=min(3, pywt.dwt_max_level(len(fp_np), pywt.Wavelet("db1").dec_len)),
        )
        num_coeffs = len(coeffs)
        keep_coeffs = max(1, int(num_coeffs * self.low_freq_ratio))
        for i in range(keep_coeffs, num_coeffs):
            coeffs[i] = np.zeros_like(coeffs[i])
        enhanced = pywt.waverec(coeffs, "db1")[: len(fp_np)]
        enhanced_tensor = torch.from_numpy(enhanced).float().to(self.device)
        return enhanced_tensor / (enhanced_tensor.norm() + 1e-8)

    def relocate_fingerprint(
        self,
        model: nn.Module,
        fingerprint: torch.Tensor,
        data_loader: torch.utils.data.DataLoader,
        strength: float = 0.1,
    ) -> nn.Module:
        """将增强后的指纹嵌入稳定参数子空间。"""
        if fingerprint.numel() == 0:
            raise ValueError("fingerprint must be non-empty")
        scores = self.identify_stable_parameters(model, data_loader)
        stable_params = self.select_stable_parameters(scores)
        if not stable_params:
            stable_params = list(scores.keys())

        enhanced = self.spectral_enhancement(fingerprint)
        fp_idx = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in stable_params or not param.requires_grad:
                    continue
                if fp_idx >= len(enhanced):
                    fp_idx = 0
                segment_len = min(param.numel(), len(enhanced))
                segment = enhanced[fp_idx : fp_idx + segment_len]
                if len(segment) < segment_len:
                    segment = enhanced[:segment_len]
                repeated = segment.repeat(
                    (param.numel() + segment_len - 1) // segment_len
                )[: param.numel()]
                param.add_(repeated.view_as(param) * strength)
                fp_idx = (fp_idx + segment_len) % len(enhanced)
        return model
