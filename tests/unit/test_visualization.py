"""Visualization模块单元测试"""

import unittest
import tempfile
import os
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.visualization import (
    plot_training_history,
    plot_client_data_distribution,
    plot_attack_robustness,
    plot_adaptive_allocation,
)


class TestPlotTrainingHistory(unittest.TestCase):
    """测试plot_training_history函数"""

    def setUp(self):
        """创建测试数据"""
        self.history = [
            {"round": 1, "loss": 2.0, "accuracy": 0.3},
            {"round": 2, "loss": 1.5, "accuracy": 0.5},
            {"round": 3, "loss": 1.0, "accuracy": 0.7},
            {"round": 4, "loss": 0.5, "accuracy": 0.9},
        ]

    def test_plot_training_history_creates_file(self):
        """测试plot_training_history创建文件"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            plot_training_history(self.history, save_path=save_path)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_plot_training_history_with_custom_title(self):
        """测试plot_training_history使用自定义标题"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            plot_training_history(
                self.history, save_path=save_path, title="Custom Title"
            )
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


class TestPlotClientDataDistribution(unittest.TestCase):
    """测试plot_client_data_distribution函数"""

    def setUp(self):
        """创建测试数据"""
        self.distributions = [
            [0.5, 0.3, 0.2, 0.0, 0.0],  # Client 0
            [0.2, 0.4, 0.3, 0.1, 0.0],  # Client 1
            [0.1, 0.1, 0.3, 0.3, 0.2],  # Client 2
            [0.0, 0.0, 0.2, 0.4, 0.4],  # Client 3
        ]

    def test_plot_client_data_distribution_creates_file(self):
        """测试plot_client_data_distribution创建文件"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            plot_client_data_distribution(self.distributions, save_path=save_path)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_plot_client_data_distribution_with_custom_title(self):
        """测试plot_client_data_distribution使用自定义标题"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            plot_client_data_distribution(
                self.distributions, save_path=save_path, title="Custom Distribution"
            )
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


class TestPlotAttackRobustness(unittest.TestCase):
    """测试plot_attack_robustness函数"""

    def setUp(self):
        """创建测试数据"""
        self.attack_results = {
            "Fine-tuning": 0.85,
            "Pruning": 0.75,
            "Quantization": 0.80,
            "Overwriting": 0.90,
            "Ambiguity": 0.95,
        }

    def test_plot_attack_robustness_creates_file(self):
        """测试plot_attack_robustness创建文件"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            plot_attack_robustness(self.attack_results, save_path=save_path)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_plot_attack_robustness_with_custom_title(self):
        """测试plot_attack_robustness使用自定义标题"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            plot_attack_robustness(
                self.attack_results, save_path=save_path, title="Custom Robustness"
            )
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


class TestPlotAdaptiveAllocation(unittest.TestCase):
    """测试plot_adaptive_allocation函数"""

    def setUp(self):
        """创建测试数据"""
        self.allocations = {
            0: [0.1, 0.12, 0.15, 0.18, 0.20],
            1: [0.15, 0.14, 0.13, 0.12, 0.11],
            2: [0.08, 0.09, 0.10, 0.11, 0.12],
        }

    def test_plot_adaptive_allocation_creates_file(self):
        """测试plot_adaptive_allocation创建文件"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            plot_adaptive_allocation(self.allocations, save_path=save_path)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_plot_adaptive_allocation_with_custom_title(self):
        """测试plot_adaptive_allocation使用自定义标题"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            plot_adaptive_allocation(
                self.allocations, save_path=save_path, title="Custom Allocation"
            )
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


if __name__ == "__main__":
    unittest.main()
