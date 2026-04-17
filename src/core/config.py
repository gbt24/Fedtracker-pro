"""配置管理系统。

本文件属于 FedTracker-Pro 项目
功能: 定义分模块配置结构并支持 YAML 读写
依赖: os, yaml, dataclasses

代码生成来源: code_generation_guide.md
章节: 阶段2 配置系统
生成日期: 2026-04-16
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class FederatedConfig:
    """联邦学习配置。"""

    num_clients: int = 50
    client_fraction: float = 0.1
    global_rounds: int = 200
    local_epochs: int = 5
    local_batch_size: int = 64
    local_lr: float = 0.01
    global_lr: float = 1.0
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0001


@dataclass
class DataConfig:
    """数据配置。"""

    dataset: str = "cifar10"
    data_dir: str = "./data"
    iid: bool = False
    alpha: float = 0.5
    num_shards: int = 200


@dataclass
class ModelConfig:
    """模型配置。"""

    name: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = False


@dataclass
class WatermarkConfig:
    """水印配置。"""

    enabled: bool = True
    type: str = "cl_backdoor"
    trigger_size: int = 100
    watermark_epochs: int = 5
    watermark_lr: float = 0.001
    cl_memory_size: int = 100
    gem_margin: float = 0.5


@dataclass
class FingerprintConfig:
    """指纹配置。"""

    enabled: bool = True
    type: str = "parametric"
    fingerprint_dim: int = 128
    embedding_strength: float = 0.1
    min_strength: float = 0.05
    identification_threshold: float = 0.5


@dataclass
class AdaptiveConfig:
    """自适应分配配置。"""

    enabled: bool = True
    evaluation_period: int = 10
    beta: float = 0.1
    min_allocation: float = 0.05


@dataclass
class CryptoConfig:
    """密码学配置。"""

    enabled: bool = True
    scheme: str = "ecdsa"
    key_size: int = 256
    hash_algorithm: str = "sha256"


@dataclass
class UnlearningConfig:
    """遗忘增强配置。"""

    enabled: bool = True
    simulation_steps: int = 100
    stability_threshold: float = 0.8
    low_freq_ratio: float = 0.8


@dataclass
class VerificationConfig:
    """验证配置。"""

    level1_threshold: float = 0.75
    level3_threshold: float = 0.9
    cross_verify_client_id: bool = True


@dataclass
class SystemConfig:
    """系统配置。"""

    device: str = "cuda"
    num_workers: int = 4
    persistent_workers: bool = False
    prefetch_factor: int = 2
    seed: int = 42
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 10


@dataclass
class ExperimentConfig:
    """实验配置。"""

    name: str = "experiment"
    tags: list[str] = field(default_factory=list)
    notes: str = ""


class Config:
    """全局配置类。"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.federated = FederatedConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.watermark = WatermarkConfig()
        self.fingerprint = FingerprintConfig()
        self.adaptive = AdaptiveConfig()
        self.crypto = CryptoConfig()
        self.unlearning = UnlearningConfig()
        self.verification = VerificationConfig()
        self.system = SystemConfig()
        self.experiment = ExperimentConfig()

        if config_path is not None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            self.load_from_yaml(config_path)

    def load_from_yaml(self, path: str) -> None:
        """从 YAML 文件加载配置。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "federated" in data:
            self.federated = FederatedConfig(**data["federated"])
        if "data" in data:
            self.data = DataConfig(**data["data"])
        if "model" in data:
            self.model = ModelConfig(**data["model"])
        if "watermark" in data:
            self.watermark = WatermarkConfig(**data["watermark"])
        if "fingerprint" in data:
            self.fingerprint = FingerprintConfig(**data["fingerprint"])
        adaptive_data = data.get("adaptive_allocation", data.get("adaptive"))
        if adaptive_data is not None:
            self.adaptive = AdaptiveConfig(**adaptive_data)
        if "crypto" in data:
            self.crypto = CryptoConfig(**data["crypto"])
        if "unlearning" in data:
            self.unlearning = UnlearningConfig(**data["unlearning"])
        if "verification" in data:
            self.verification = VerificationConfig(**data["verification"])
        if "system" in data:
            self.system = SystemConfig(**data["system"])
        if "experiment" in data:
            self.experiment = ExperimentConfig(**data["experiment"])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "federated": self.federated.__dict__,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "watermark": self.watermark.__dict__,
            "fingerprint": self.fingerprint.__dict__,
            "adaptive_allocation": self.adaptive.__dict__,
            "crypto": self.crypto.__dict__,
            "unlearning": self.unlearning.__dict__,
            "verification": self.verification.__dict__,
            "system": self.system.__dict__,
            "experiment": self.experiment.__dict__,
        }

    def save_to_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """获取全局配置实例。"""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def set_config(config: Config) -> None:
    """设置全局配置实例。"""
    global _global_config
    _global_config = config
