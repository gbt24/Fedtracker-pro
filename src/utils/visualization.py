"""可视化工具"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import os


def plot_training_history(
    history: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Training History",
):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    rounds = [h["round"] for h in history]

    # 损失曲线
    if "loss" in history[0]:
        losses = [h["loss"] for h in history]
        axes[0].plot(rounds, losses, label="Loss")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True)

    # 准确率曲线
    if "accuracy" in history[0]:
        accuracies = [h["accuracy"] for h in history]
        axes[1].plot(rounds, accuracies, label="Accuracy", color="orange")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_client_data_distribution(
    distributions: List[np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Client Data Distribution",
):
    """绘制客户端数据分布热图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 转换为矩阵
    dist_matrix = np.array(distributions)

    sns.heatmap(
        dist_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=[f"Class {i}" for i in range(dist_matrix.shape[1])],
        yticklabels=[f"Client {i}" for i in range(dist_matrix.shape[0])],
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_attack_robustness(
    attack_results: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Attack Robustness",
):
    """绘制攻击鲁棒性柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    attacks = list(attack_results.keys())
    values = list(attack_results.values())

    colors = ["green" if v > 0.8 else "orange" if v > 0.5 else "red" for v in values]

    bars = ax.barh(attacks, values, color=colors)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Survival Rate")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.01, i, f"{val:.2%}", va="center")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_adaptive_allocation(
    allocations: Dict[int, List[float]],
    save_path: Optional[str] = None,
    title: str = "Adaptive Allocation Over Time",
):
    """绘制自适应分配变化"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for client_id, history in allocations.items():
        rounds = list(range(len(history)))
        ax.plot(rounds, history, label=f"Client {client_id}", marker="o")

    ax.set_xlabel("Evaluation Period")
    ax.set_ylabel("Allocation Weight")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
