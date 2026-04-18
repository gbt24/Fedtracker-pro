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
import math
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
        self.crypto_verifiers: Dict[int, CryptographicVerification] = {}
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
                pin_memory=self.config.system.pin_memory,
                persistent_workers=self.config.system.persistent_workers,
                prefetch_factor=self.config.system.prefetch_factor,
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
            watermark_type = self.config.watermark.type.lower()
            if watermark_type not in {"cl_backdoor", "cl", "continual_learning"}:
                raise ValueError(
                    f"Unsupported watermark type: {self.config.watermark.type}"
                )
            self.watermarker = ContinualLearningWatermark(
                trigger_size=self.config.watermark.trigger_size,
                target_label=0,
                device=self.device,
                memory_size=self.config.watermark.cl_memory_size,
                gem_margin=self.config.watermark.gem_margin,
            )

        if self.config.fingerprint.enabled:
            fingerprint_type = self.config.fingerprint.type.lower()
            if fingerprint_type not in {"parametric", "param"}:
                raise ValueError(
                    f"Unsupported fingerprint type: {self.config.fingerprint.type}"
                )
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
            self.crypto_verifiers = {}

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
        self.crypto_verifiers = {}
        for client_id in range(self.config.federated.num_clients):
            client_crypto_verifier: Optional[CryptographicVerification] = None
            if self.config.crypto.enabled:
                client_crypto_verifier = CryptographicVerification(
                    key_size=self.config.crypto.key_size,
                    scheme=self.config.crypto.scheme,
                    hash_algorithm=self.config.crypto.hash_algorithm,
                    device=self.device,
                )
                self.crypto_verifiers[client_id] = client_crypto_verifier

            if self.fingerprint_registry is not None:
                fp = self.fingerprint_registry.get_fingerprint(client_id)
                client = ProtectedClient(
                    client_id=client_id,
                    model=copy.deepcopy(self.global_model),
                    train_loader=None,
                    fingerprinter=fp,
                    crypto_verifier=client_crypto_verifier,
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
                    train_loader=None,
                    device=self.device,
                    local_epochs=self.config.federated.local_epochs,
                    local_lr=self.config.federated.local_lr,
                    optimizer_name=self.config.federated.optimizer,
                    momentum=self.config.federated.momentum,
                    weight_decay=self.config.federated.weight_decay,
                )
            self.clients.append(client)

    def _verify_crypto_model(
        self,
        model: nn.Module,
        *,
        preferred_client_id: Optional[int] = None,
        candidate_clients: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        """使用客户端签名上下文验证模型。"""
        candidates: List[int] = []
        if preferred_client_id is not None:
            candidates.append(preferred_client_id)
        if candidate_clients is not None:
            candidates.extend(candidate_clients)
        if not candidates:
            candidates = list(range(len(self.clients)))

        tried: set[int] = set()
        for client_id in candidates:
            if client_id in tried:
                continue
            tried.add(client_id)
            verifier = self.crypto_verifiers.get(client_id)
            if verifier is None:
                continue
            try:
                result = verifier.verify_model(model)
            except Exception:
                continue
            payload: Dict[str, object] = dict(result)
            payload.setdefault("client_id", client_id)
            if bool(payload.get("is_valid", False)):
                return payload

        if self.crypto_verifier is not None:
            return self.crypto_verifier.verify_model(model)
        return {"is_valid": False}

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

        protection_strengths: Dict[int, float] = {
            cid: max(
                self.config.fingerprint.embedding_strength,
                self.config.fingerprint.min_strength,
            )
            for cid in range(self.config.federated.num_clients)
        }

        for round_idx in rounds:
            local_states: list[Dict[str, torch.Tensor]] = []
            global_state = self.server.get_global_state(to_cpu=False)
            selected_clients = self._select_clients()
            tolerance_scores: Dict[str, float] = {}
            for client_id in selected_clients:
                train_loader = self.data_manager.get_client_loader(
                    client_id=client_id,
                    batch_size=self.config.federated.local_batch_size,
                    shuffle=True,
                )

                strength = protection_strengths.get(
                    client_id,
                    max(
                        self.config.fingerprint.embedding_strength,
                        self.config.fingerprint.min_strength,
                    ),
                )
                client = self.clients[client_id]
                if isinstance(client, ProtectedClient):
                    local_state = client.local_train(
                        global_state=global_state,
                        return_cpu_state=True,
                        train_loader=train_loader,
                        protection_strength=strength,
                        unlearning_guide=self.unlearning_guide,
                    )
                else:
                    local_state = client.local_train(
                        global_state=global_state,
                        return_cpu_state=True,
                        train_loader=train_loader,
                    )

                if self.watermarker is not None and hasattr(self.watermarker, "embed"):
                    try:
                        if hasattr(self.watermarker, "reset_memory"):
                            self.watermarker.reset_memory()
                        wm_model = self.watermarker.embed(
                            self.clients[client_id].model,
                            train_loader,
                            epochs=self.config.watermark.watermark_epochs,
                            lr=self.config.watermark.watermark_lr,
                        )
                        if wm_model is not None:
                            self.clients[client_id].model = wm_model.to(
                                self.clients[client_id].device
                            )

                        embed_fn = getattr(
                            self.clients[client_id], "embed_protection", None
                        )
                        if callable(embed_fn):
                            embed_fn(
                                train_loader=train_loader,
                                fingerprint_strength=strength,
                                unlearning_guide=self.unlearning_guide,
                            )

                        local_state = self.clients[client_id].get_model_state(
                            to_cpu=True
                        )
                    except Exception as exc:
                        self.logger.warning(
                            f"Watermark embedding failed for client {client_id}: {exc}"
                        )

                if (
                    self.config.crypto.enabled
                    and not isinstance(client, ProtectedClient)
                    and client_id in self.crypto_verifiers
                ):
                    self.crypto_verifiers[client_id].embed_to_model(
                        client.model,
                        client_id=client_id,
                    )
                    local_state = client.get_model_state(to_cpu=True)

                local_states.append(local_state)

                if (
                    self.adaptive_allocator is not None
                    and self.fingerprint_registry is not None
                ):
                    try:
                        fingerprinter = self.fingerprint_registry.get_fingerprint(
                            client_id
                        )
                        similarity = fingerprinter.verify(self.clients[client_id].model)
                    except Exception:
                        similarity = 0.0
                    latest = self.clients[client_id].training_history[-1]
                    accuracy = float(latest.get("accuracy", 0.0)) / 100.0
                    loss = float(latest.get("loss", 0.0))
                    tolerance_scores[str(client_id)] = (
                        self.adaptive_allocator.evaluate_tolerance(
                            accuracy=accuracy,
                            loss=loss,
                            fingerprint_similarity=similarity,
                        )
                    )

            self.server.aggregate(local_states)
            self.global_model = self.server.global_model

            if (
                self.adaptive_allocator is not None
                and tolerance_scores
                and (round_idx + 1) % self.adaptive_allocator.evaluation_period == 0
            ):
                allocations = self.adaptive_allocator.allocate(tolerance_scores)
                for client_key, allocation in allocations.items():
                    client_id = int(client_key)
                    base_strength = max(
                        self.config.fingerprint.embedding_strength,
                        self.config.fingerprint.min_strength,
                    )
                    adaptive_ratio = float(allocation) / max(
                        self.adaptive_allocator.beta,
                        1e-8,
                    )
                    adaptive_ratio = max(0.0, min(1.0, adaptive_ratio))
                    protection_strengths[client_id] = max(
                        self.config.fingerprint.min_strength,
                        self.config.fingerprint.min_strength
                        + (base_strength - self.config.fingerprint.min_strength)
                        * adaptive_ratio,
                    )

            if (round_idx + 1) % self.config.system.save_frequency == 0:
                self.server.evaluate(self.data_manager.get_test_loader())
                self._save_checkpoint(round_idx + 1)

    def _ensure_watermark_trigger_set(self) -> bool:
        """确保水印触发集可用于验证阶段。"""
        if self.watermarker is None:
            return False
        if not hasattr(self.watermarker, "trigger_set"):
            return False
        if getattr(self.watermarker, "trigger_set") is not None:
            return True
        if not hasattr(self.watermarker, "generate_trigger_set"):
            return False

        try:
            loader = None
            if self.data_manager is not None:
                loader = self.data_manager.get_test_loader()
            self.watermarker.generate_trigger_set(loader)
        except Exception as exc:
            self.logger.warning(f"Failed to generate watermark trigger set: {exc}")
            return False
        return True

    def verify_ownership(
        self,
        suspicious_model: nn.Module,
        candidate_clients: Optional[List[int]] = None,
        enforce_crypto: bool = True,
        crypto_result: Optional[Dict[str, object]] = None,
        watermark_accuracy: Optional[float] = None,
        enforce_watermark: bool = True,
        level1_threshold_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[int], float]:
        """执行三层验证并返回所有权判断。

        Returns:
            (是否属于本 FL 系统, 泄露者 client_id, 置信度)
        """
        if candidate_clients is None:
            candidate_clients = list(range(len(self.clients)))
        if not candidate_clients:
            return False, None, 0.0

        matched_id: Optional[int] = None
        similarity = 1.0

        if self.fingerprint_registry is not None:
            if level1_threshold_override is not None and (
                not math.isfinite(level1_threshold_override)
                or level1_threshold_override < -1.0
                or level1_threshold_override > 1.0
            ):
                raise ValueError(
                    "level1_threshold_override must be finite and in [-1, 1]"
                )

            matched_id, similarity = self.fingerprint_registry.identify_client(
                suspicious_model, candidate_clients
            )
            if matched_id == -1:
                matched_id = None
            level1_threshold = (
                self.config.verification.level1_threshold
                if level1_threshold_override is None
                else level1_threshold_override
            )
            if similarity < level1_threshold:
                return False, None, 0.0

        if self.crypto_verifier is not None or self.crypto_verifiers:
            local_crypto_result: Dict[str, object] = (
                {"is_valid": False} if crypto_result is None else dict(crypto_result)
            )
            if crypto_result is None:
                try:
                    local_crypto_result = self._verify_crypto_model(
                        suspicious_model,
                        preferred_client_id=matched_id,
                        candidate_clients=candidate_clients,
                    )
                except Exception:
                    if enforce_crypto:
                        return False, None, similarity

            if enforce_crypto and not bool(local_crypto_result.get("is_valid", False)):
                return False, None, similarity

            signed_client = local_crypto_result.get("client_id")
            if signed_client is not None:
                signed_client = int(signed_client)
                if matched_id is None:
                    matched_id = signed_client
            if (
                enforce_crypto
                and self.config.verification.cross_verify_client_id
                and signed_client is not None
                and matched_id is not None
                and matched_id != signed_client
            ):
                return False, None, similarity

        if matched_id is None and candidate_clients:
            matched_id = candidate_clients[0]

        if self.watermarker is not None:
            wm_accuracy = watermark_accuracy
            if wm_accuracy is None:
                if not self._ensure_watermark_trigger_set():
                    if enforce_watermark:
                        return False, matched_id, float((similarity + 1.0) / 2)
                    return True, matched_id, float((similarity + 1.0) / 2)

                try:
                    wm_accuracy = self.watermarker.verify(suspicious_model)
                except Exception as exc:
                    self.logger.warning(f"Watermark verification failed: {exc}")
                    if enforce_watermark:
                        return False, matched_id, float((similarity + 1.0) / 2)
                    return True, matched_id, float((similarity + 1.0) / 2)

            if not math.isfinite(float(wm_accuracy)):
                if enforce_watermark:
                    return False, matched_id, float((similarity + 1.0) / 2)
                return True, matched_id, float((similarity + 1.0) / 2)

            if wm_accuracy < self.config.verification.level3_threshold:
                if enforce_watermark:
                    return False, matched_id, float((similarity + 1.0) / 2)
                return True, matched_id, float((similarity + 1.0) / 2)
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
        try:
            checkpoint = torch.load(path, map_location="cpu")
            checkpoint["crypto_state"] = {
                cid: verifier.export_state()
                for cid, verifier in self.crypto_verifiers.items()
            }
            torch.save(checkpoint, path)
        except Exception as exc:
            self.logger.warning(f"Failed to persist crypto state: {exc}")

    def load_checkpoint(self, path: str) -> None:
        """加载检查点并恢复可用的密码学状态。"""
        if self.server is None:
            raise RuntimeError("Server is not initialized")
        self.server.load_checkpoint(path)
        self.global_model = self.server.global_model

        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as exc:
            self.logger.warning(f"Failed to load checkpoint payload: {exc}")
            return

        crypto_state = checkpoint.get("crypto_state")
        if not isinstance(crypto_state, dict):
            return

        for key, state in crypto_state.items():
            try:
                client_id = int(key)
            except Exception:
                continue
            verifier = self.crypto_verifiers.get(client_id)
            if verifier is None:
                continue
            try:
                verifier.load_state(state)
            except Exception as exc:
                self.logger.warning(
                    f"Failed to restore crypto state for client {client_id}: {exc}"
                )

    def _select_robustness_victim_client(self) -> int:
        """选择用于鲁棒性评估的受害客户端。

        优先选择“自指纹相似度”最高的客户端，避免在少轮训练时固定
        client 0 恰好未被采样导致结果全零。
        """
        if not self.clients:
            raise RuntimeError("No clients available for robustness evaluation")

        if self.fingerprint_registry is None:
            return 0

        best_id = 0
        best_score = -2.0
        for client_id, client in enumerate(self.clients):
            try:
                fp = self.fingerprint_registry.get_fingerprint(client_id)
                score = fp.verify(client.model)
            except Exception:
                score = -2.0

            if score > best_score:
                best_score = score
                best_id = client_id

        return best_id

    def evaluate_attack_robustness(
        self,
        attacks: List,
        test_loader: torch.utils.data.DataLoader,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        enforce_crypto: bool = True,
        enforce_watermark: bool = False,
        level1_threshold_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """评估攻击下的所有权存活率。"""
        _ = test_loader
        if not self.clients:
            raise RuntimeError("No clients available for robustness evaluation")

        results: Dict[str, float] = {}
        victim_client_id = self._select_robustness_victim_client()
        victim_model_base = self.clients[victim_client_id].model
        attack_iter = attacks
        if show_progress:
            attack_iter = tqdm(
                attacks,
                total=len(attacks),
                desc=progress_desc or "attack-eval",
                unit="attack",
            )

        for attack in attack_iter:
            victim_model = copy.deepcopy(victim_model_base)
            kwargs: Dict[str, object] = {}
            signature = inspect.signature(attack.attack)
            if "train_loader" in signature.parameters:
                kwargs["train_loader"] = test_loader
            if "query_loader" in signature.parameters:
                kwargs["query_loader"] = test_loader

            attacked_model = attack.attack(victim_model, **kwargs)
            crypto_result: Optional[Dict[str, object]] = None
            watermark_accuracy: Optional[float] = None
            if self.crypto_verifier is not None or self.crypto_verifiers:
                try:
                    crypto_result = self._verify_crypto_model(
                        attacked_model,
                        preferred_client_id=victim_client_id,
                        candidate_clients=[victim_client_id],
                    )
                except Exception:
                    crypto_result = {"is_valid": False}

            if self.watermarker is not None and self._ensure_watermark_trigger_set():
                try:
                    watermark_accuracy = self.watermarker.verify(attacked_model)
                except Exception:
                    watermark_accuracy = None

            local_level1_threshold = level1_threshold_override
            if (
                local_level1_threshold is None
                and self.fingerprint_registry is not None
                and not enforce_crypto
                and not enforce_watermark
            ):
                local_level1_threshold = min(
                    self.config.verification.level1_threshold,
                    self.fingerprint_registry.identification_threshold,
                )

            is_owner, _, _ = self.verify_ownership(
                attacked_model,
                enforce_crypto=enforce_crypto,
                enforce_watermark=enforce_watermark,
                crypto_result=crypto_result,
                watermark_accuracy=watermark_accuracy,
                level1_threshold_override=local_level1_threshold,
            )
            attack_name = attack.get_attack_name()
            results[attack_name] = 1.0 if is_owner else 0.0

            if self.crypto_verifier is not None or self.crypto_verifiers:
                crypto_valid = bool((crypto_result or {}).get("is_valid", False))
                results[f"{attack_name}_crypto_pass_rate"] = (
                    1.0 if crypto_valid else 0.0
                )

            if self.watermarker is not None:
                watermark_valid = (
                    watermark_accuracy is not None
                    and watermark_accuracy >= self.config.verification.level3_threshold
                )
                results[f"{attack_name}_watermark_pass_rate"] = (
                    1.0 if watermark_valid else 0.0
                )

            if self.fingerprint_registry is not None:
                _, similarity = self.fingerprint_registry.identify_client(
                    attacked_model,
                    list(range(len(self.clients))),
                )
                results[f"{attack_name}_fingerprint_similarity"] = float(similarity)
        return results
