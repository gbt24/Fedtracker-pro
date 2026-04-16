"""客户端指纹注册表。

本文件属于 FedTracker-Pro 项目
功能: 为每个客户端生成唯一指纹并支持溯源匹配
依赖: torch, torch.nn, src.defense.fingerprint.param_fingerprint

代码生成来源: per_client_traceability_plan.md
章节: 阶段A
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .param_fingerprint import ParametricFingerprint


class ClientFingerprintRegistry:
    """管理所有客户端的唯一指纹，支持溯源匹配。"""

    def __init__(
        self,
        fingerprint_dim: int = 128,
        embedding_strength: float = 0.1,
        min_strength: float = 0.05,
        device: str = "cpu",
        base_seed: int = 42,
        identification_threshold: float = 0.5,
    ) -> None:
        if fingerprint_dim <= 0:
            raise ValueError("fingerprint_dim must be greater than 0")
        if identification_threshold < -1.0 or identification_threshold > 1.0:
            raise ValueError("identification_threshold must be in [-1, 1]")
        self.fingerprint_dim = fingerprint_dim
        self.embedding_strength = embedding_strength
        self.min_strength = min_strength
        self.device = device
        self.base_seed = base_seed
        self.identification_threshold = identification_threshold
        self._registry: Dict[int, ParametricFingerprint] = {}

    def register_client(self, client_id: int) -> ParametricFingerprint:
        """为客户端注册唯一指纹（seed = base_seed + client_id）。"""
        if client_id in self._registry:
            raise ValueError(f"Client {client_id} already registered")
        fp = ParametricFingerprint(
            fingerprint_dim=self.fingerprint_dim,
            embedding_strength=self.embedding_strength,
            min_strength=self.min_strength,
            device=self.device,
            seed=self.base_seed + client_id,
        )
        fp.generate()
        self._registry[client_id] = fp
        return fp

    def register_clients(self, client_ids: List[int]) -> None:
        """批量注册客户端指纹。"""
        for cid in client_ids:
            self.register_client(cid)

    def get_fingerprint(self, client_id: int) -> ParametricFingerprint:
        """获取指定客户端的指纹实例。"""
        if client_id not in self._registry:
            raise KeyError(f"Client {client_id} not registered")
        return self._registry[client_id]

    @property
    def registered_ids(self) -> List[int]:
        """返回已注册的客户端 ID 列表。"""
        return sorted(self._registry.keys())

    def embed_client_fingerprint(self, client_id: int, model: nn.Module) -> nn.Module:
        """将指定客户端的指纹嵌入模型。"""
        fp = self.get_fingerprint(client_id)
        return fp.embed(model)

    def identify_client(
        self,
        suspicious_model: nn.Module,
        candidate_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """从可疑模型中提取指纹，匹配最可能的客户端。

        Returns:
            (client_id, similarity) — 若最高相似度低于阈值则 client_id = -1
        """
        if not self._registry:
            return -1, 0.0

        ids = (
            candidate_ids if candidate_ids is not None else list(self._registry.keys())
        )
        best_id = -1
        best_score = -2.0

        for cid in ids:
            if cid not in self._registry:
                continue
            score = self._registry[cid].verify(suspicious_model)
            if score > best_score:
                best_score = score
                best_id = cid

        if best_score < self.identification_threshold:
            return -1, best_score

        return best_id, best_score

    def get_all_similarities(self, model: nn.Module) -> Dict[int, float]:
        """返回模型与所有已注册客户端指纹的相似度字典。"""
        return {cid: fp.verify(model) for cid, fp in self._registry.items()}
