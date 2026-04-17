"""FedTracker-Pro 主框架。

本文件属于 FedTracker-Pro 项目
功能: 组装联邦训练流程、三层验证与攻击鲁棒性评估
依赖: src.core.config, src.core.base_client, src.core.base_server

代码生成来源: code_generation_guide.md
章节: 阶段6 主框架类
生成日期: 2026-04-16
"""

from __future__ import annotations

import copy
import inspect
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):
        _ = args, kwargs
        return iterable


from .base_client import StandardClient
from .base_server import BaseServer
from .config import Config
from .protected_client import ProtectedClient
from ..aggregation.fed_avg import FedAvgAggregator
from ..datasets.federated_dataset import FederatedDataManager
from ..defense.adaptive_allocation import AdaptiveAllocator
from ..defense.crypto_verification import CryptographicVerification
from ..defense.fingerprint.client_fingerprint_registry import ClientFingerprintRegistry
from ..defense.unlearning_guided import UnlearningGuidedRelocation
from ..defense.watermark.cl_watermark import ContinualLearningWatermark
from ..utils.logger import get_logger


class FedTrackerPro:
    """FedTracker-Pro 主框架。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = (
            config.system.device
            if torch.cuda.is_available() and config.system.device.startswith("cuda")
            else "cpu"
        )
        self.logger = get_logger(
            name="FedTrackerPro",
            log_dir=config.system.log_dir,
            level=20,
        )

        self.data_manager: Optional[FederatedDataManager] = None
        self.global_model: Optional[nn.Module] = None
        self.server: Optional[BaseServer] = None
        self.clients: list[StandardClient] = []
        self.aggregator: Optional[FedAvgAggregator] = None

        self.watermarker: Optional[ContinualLearningWatermark] = None
        self.fingerprint_registry: Optional[ClientFingerprintRegistry] = None
        self.adaptive_allocator: Optional[AdaptiveAllocator] = None
        self.crypto_verifier: Optional[CryptographicVerification] = None
        self.unlearning_guide: Optional[UnlearningGuidedRelocation] = None

    def initialize(
        self,
        model: nn.Module,
        data_manager: Optional[FederatedDataManager] = None,
    ) -> None:
        """初始化框架组件。"""
        if data_manager is None:
            self.data_manager = FederatedDataManager(
                dataset_name=self.config.data.dataset,
                data_dir=self.config.data.data_dir,
                num_clients=self.config.federated.num_clients,
                iid=self.config.data.iid,
                alpha=self.config.data.alpha,
                num_shards=self.config.data.num_shards,
                num_workers=self.config.system.num_workers,
                pin_memory=self.device.startswith("cuda"),
            )
        else:
            self.data_manager = data_manager

        self.aggregator = FedAvgAggregator(device=self.device)
        self.server = BaseServer(
            model=model,
            aggregator=self.aggregator,
            device=self.device,
            global_lr=self.config.federated.global_lr,
        )
        self.global_model = self.server.global_model

        self._setup_defense_modules()
        self._create_clients()

    def _setup_defense_modules(self) -> None:
        """按配置启用防御模块。"""
        if self.config.watermark.enabled:
            self.watermarker = ContinualLearningWatermark(
                trigger_size=self.config.watermark.trigger_size,
                target_label=0,
                device=self.device,
                memory_size=self.config.watermark.cl_memory_size,
                gem_margin=self.config.watermark.gem_margin,
            )

        if self.config.fingerprint.enabled:
            self.fingerprint_registry = ClientFingerprintRegistry(
                fingerprint_dim=self.config.fingerprint.fingerprint_dim,
                embedding_strength=self.config.fingerprint.embedding_strength,
                min_strength=self.config.fingerprint.min_strength,
                device=self.device,
                base_seed=self.config.system.seed,
                identification_threshold=self.config.fingerprint.identification_threshold,
            )
            self.fingerprint_registry.register_clients(
                list(range(self.config.federated.num_clients))
            )

        if self.config.adaptive.enabled:
            self.adaptive_allocator = AdaptiveAllocator(
                beta=self.config.adaptive.beta,
                min_allocation=self.config.adaptive.min_allocation,
                evaluation_period=self.config.adaptive.evaluation_period,
                device=self.device,
            )

        if self.config.crypto.enabled:
            self.crypto_verifier = CryptographicVerification(
                key_size=self.config.crypto.key_size,
                scheme=self.config.crypto.scheme,
                hash_algorithm=self.config.crypto.hash_algorithm,
                device=self.device,
            )

        if self.config.unlearning.enabled:
            self.unlearning_guide = UnlearningGuidedRelocation(
                simulation_steps=self.config.unlearning.simulation_steps,
                stability_threshold=self.config.unlearning.stability_threshold,
                low_freq_ratio=self.config.unlearning.low_freq_ratio,
                device=self.device,
            )

    def _create_clients(self) -> None:
        """创建联邦客户端。"""
        if self.data_manager is None or self.global_model is None:
            raise RuntimeError("Framework not initialized")

        self.clients = []
        for client_id in range(self.config.federated.num_clients):
            train_loader = self.data_manager.get_client_loader(
                client_id=client_id,
                batch_size=self.config.federated.local_batch_size,
                shuffle=True,
            )

            if self.fingerprint_registry is not None:
                fp = self.fingerprint_registry.get_fingerprint(client_id)
                client = ProtectedClient(
                    client_id=client_id,
                    model=copy.deepcopy(self.global_model),
                    train_loader=train_loader,
                    fingerprinter=fp,
                    crypto_verifier=self.crypto_verifier,
                    device=self.device,
                    local_epochs=self.config.federated.local_epochs,
                    local_lr=self.config.federated.local_lr,
                    optimizer_name=self.config.federated.optimizer,
                    momentum=self.config.federated.momentum,
                    weight_decay=self.config.federated.weight_decay,
                )
            else:
                client = StandardClient(
                    client_id=client_id,
                    model=copy.deepcopy(self.global_model),
                    train_loader=train_loader,
                    device=self.device,
                    local_epochs=self.config.federated.local_epochs,
                    local_lr=self.config.federated.local_lr,
                    optimizer_name=self.config.federated.optimizer,
                    momentum=self.config.federated.momentum,
                    weight_decay=self.config.federated.weight_decay,
                )
            self.clients.append(client)

    def _select_clients(self) -> list[int]:
        """按采样率随机选择客户端。"""
        num_clients = self.config.federated.num_clients
        fraction = self.config.federated.client_fraction
        num_selected = max(1, int(num_clients * fraction))
        return torch.randperm(num_clients)[:num_selected].tolist()

    def train(
        self,
        num_rounds: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> None:
        """执行联邦训练。"""
        if self.server is None:
            raise RuntimeError("Framework not initialized")
        if self.data_manager is None:
            raise RuntimeError("Data manager is not initialized")

        if num_rounds is None:
            num_rounds = self.config.federated.global_rounds

        rounds = range(num_rounds)
        if show_progress:
            rounds = tqdm(
                rounds,
                total=num_rounds,
                desc=progress_desc or "federated-train",
                unit="round",
            )

        for round_idx in rounds:
            local_states: list[Dict[str, torch.Tensor]] = []
            global_state = self.server.get_global_state(to_cpu=False)
            for client_id in self._select_clients():
                local_state = self.clients[client_id].local_train(
                    global_state=global_state,
                    return_cpu_state=True,
                )
                local_states.append(local_state)

            self.server.aggregate(local_states)
            self.global_model = self.server.global_model

            if (round_idx + 1) % self.config.system.save_frequency == 0:
                self.server.evaluate(self.data_manager.get_test_loader())
                self._save_checkpoint(round_idx + 1)

    def verify_ownership(
        self,
        suspicious_model: nn.Module,
        candidate_clients: Optional[List[int]] = None,
    ) -> Tuple[bool, Optional[int], float]:
        """执行三层验证并返回所有权判断。

        Returns:
            (是否属于本 FL 系统, 泄露者 client_id, 置信度)
        """
        if candidate_clients is None:
            candidate_clients = list(range(len(self.clients)))
        if not candidate_clients:
            return False, None, 0.0

        matched_id: Optional[int] = candidate_clients[0] if candidate_clients else None
        similarity = 1.0

        if self.fingerprint_registry is not None:
            matched_id, similarity = self.fingerprint_registry.identify_client(
                suspicious_model, candidate_clients
            )
            if similarity < self.config.verification.level1_threshold:
                return False, None, 0.0

        if self.crypto_verifier is not None:
            crypto_result = self.crypto_verifier.verify_model(suspicious_model)
            if not bool(crypto_result.get("is_valid", False)):
                return False, None, similarity
            signed_client = crypto_result.get("client_id")
            if (
                self.config.verification.cross_verify_client_id
                and signed_client is not None
                and matched_id is not None
                and matched_id != signed_client
            ):
                return False, None, similarity

        if self.watermarker is not None:
            wm_accuracy = self.watermarker.verify(suspicious_model)
            if wm_accuracy < self.config.verification.level3_threshold:
                return False, matched_id, (similarity + 1.0) / 2
            confidence = (similarity + 1.0 + wm_accuracy) / 3
            return True, matched_id, float(confidence)

        return True, matched_id, float((similarity + 1.0) / 2)

    def _save_checkpoint(self, round_num: int) -> None:
        """保存训练检查点。"""
        if self.server is None:
            raise RuntimeError("Server is not initialized")
        checkpoint_dir = self.config.system.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"checkpoint_round{round_num}.pth")
        self.server.save_checkpoint(path)

    def evaluate_attack_robustness(
        self,
        attacks: List,
        test_loader: torch.utils.data.DataLoader,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> Dict[str, float]:
        """评估攻击下的所有权存活率。"""
        _ = test_loader
        if not self.clients:
            raise RuntimeError("No clients available for robustness evaluation")

        results: Dict[str, float] = {}
        attack_iter = attacks
        if show_progress:
            attack_iter = tqdm(
                attacks,
                total=len(attacks),
                desc=progress_desc or "attack-eval",
                unit="attack",
            )

        for attack in attack_iter:
            victim_model = copy.deepcopy(self.clients[0].model)
            kwargs: Dict[str, object] = {}
            signature = inspect.signature(attack.attack)
            if "train_loader" in signature.parameters:
                kwargs["train_loader"] = test_loader
            if "query_loader" in signature.parameters:
                kwargs["query_loader"] = test_loader

            attacked_model = attack.attack(victim_model, **kwargs)
            is_owner, _, _ = self.verify_ownership(attacked_model)
            results[attack.get_attack_name()] = 1.0 if is_owner else 0.0
        return results
