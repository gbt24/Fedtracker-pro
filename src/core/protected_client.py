"""受保护客户端实现。

本文件属于 FedTracker-Pro 项目
功能: 在本地训练后自动嵌入指纹和签名的客户端
依赖: torch, src.core.base_client, src.defense.fingerprint.param_fingerprint

代码生成来源: per_client_traceability_plan.md
章节: 阶段B
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .base_client import StandardClient
from ..defense.fingerprint.param_fingerprint import ParametricFingerprint


class ProtectedClient(StandardClient):
    """在本地训练后自动嵌入指纹和签名的客户端。"""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        fingerprinter: ParametricFingerprint,
        train_loader: Optional[torch.utils.data.DataLoader] = None,
        crypto_verifier: Optional[object] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cpu",
        local_epochs: int = 5,
        local_lr: float = 0.01,
        optimizer_name: str = "sgd",
        **optimizer_kwargs,
    ) -> None:
        super().__init__(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            local_epochs=local_epochs,
            local_lr=local_lr,
            optimizer_name=optimizer_name,
            **optimizer_kwargs,
        )
        self.fingerprinter = fingerprinter
        self.crypto_verifier = crypto_verifier

    def embed_protection(
        self,
        train_loader: Optional[torch.utils.data.DataLoader] = None,
        fingerprint_strength: Optional[float] = None,
        unlearning_guide: Optional[object] = None,
    ) -> None:
        """在本地训练后将指纹嵌入模型。

        流程:
        1. 调用 fingerprinter.embed(self.model) 嵌入客户端指纹
        2. 如果 crypto_verifier 存在，调用 embed_to_model 嵌入签名
        """
        original_strength = self.fingerprinter.embedding_strength
        if fingerprint_strength is not None and fingerprint_strength > 0:
            self.fingerprinter.embedding_strength = fingerprint_strength

        try:
            if (
                unlearning_guide is not None
                and train_loader is not None
                and hasattr(unlearning_guide, "relocate_fingerprint")
            ):
                if self.fingerprinter.fingerprint is None:
                    self.fingerprinter.generate()
                strength = max(
                    self.fingerprinter.embedding_strength,
                    self.fingerprinter.min_strength,
                )
                unlearning_guide.relocate_fingerprint(
                    self.model,
                    self.fingerprinter.fingerprint,
                    train_loader,
                    strength=strength,
                )
            else:
                self.fingerprinter.embed(self.model)
        finally:
            self.fingerprinter.embedding_strength = original_strength

        if self.crypto_verifier is not None and hasattr(
            self.crypto_verifier, "embed_to_model"
        ):
            self.crypto_verifier.embed_to_model(self.model, client_id=self.client_id)

    def local_train(
        self,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
        return_cpu_state: bool = True,
        train_loader: Optional[torch.utils.data.DataLoader] = None,
        protection_strength: Optional[float] = None,
        unlearning_guide: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """重写: 训练 → 嵌入保护 → 返回状态。"""
        super().local_train(
            global_state,
            return_cpu_state=False,
            train_loader=train_loader,
        )
        self.embed_protection(
            train_loader=train_loader,
            fingerprint_strength=protection_strength,
            unlearning_guide=unlearning_guide,
        )
        return self.get_model_state(to_cpu=return_cpu_state)
