"""评估指标工具"""

import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


class MetricsTracker:
    """指标追踪器"""

    def __init__(self):
        self.metrics = {}
        self.history = []

    def update(self, metrics_dict: Dict[str, float], round_num: int = None):
        """更新指标"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        if round_num is not None:
            self.history.append({"round": round_num, **metrics_dict})

    def get_latest(self, key: str) -> float:
        """获取最新指标值"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return self.metrics[key][-1]
        return 0.0

    def get_average(self, key: str) -> float:
        """获取平均指标值"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(self.metrics[key])
        return 0.0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "latest": {k: v[-1] if v else 0.0 for k, v in self.metrics.items()},
            "average": {k: np.mean(v) if v else 0.0 for k, v in self.metrics.items()},
            "history": self.history,
        }

    def save(self, filepath: str):
        """保存到文件"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> float:
    """计算模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return correct / total if total > 0 else 0.0


def compute_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str = "cuda",
) -> float:
    """计算模型损失"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def compute_fingerprint_similarity(
    fingerprint1: torch.Tensor, fingerprint2: torch.Tensor
) -> float:
    """
    计算两个指纹的相似度 (FSS)

    使用余弦相似度
    """
    return torch.nn.functional.cosine_similarity(
        fingerprint1.unsqueeze(0), fingerprint2.unsqueeze(0)
    ).item()


def compute_verification_metrics(
    predictions: List[bool], ground_truth: List[bool]
) -> Dict[str, float]:
    """计算验证指标"""
    y_true = [int(x) for x in ground_truth]
    y_pred = [int(x) for x in predictions]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
