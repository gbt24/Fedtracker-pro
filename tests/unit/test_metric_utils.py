"""MetricUtils模块单元测试"""

import unittest
import torch
import torch.nn as nn
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.metric_utils import (
    MetricsTracker,
    compute_accuracy,
    compute_loss,
    compute_fingerprint_similarity,
    compute_verification_metrics,
)


class TestMetricsTracker(unittest.TestCase):
    """测试MetricsTracker类"""

    def setUp(self):
        """每个测试前设置"""
        self.tracker = MetricsTracker()

    def test_metrics_tracker_initialization(self):
        """测试MetricsTracker初始化"""
        self.assertEqual(self.tracker.metrics, {})
        self.assertEqual(self.tracker.history, [])

    def test_metrics_tracker_update(self):
        """测试更新指标"""
        self.tracker.update({"accuracy": 0.9, "loss": 0.1}, round_num=1)
        self.assertIn("accuracy", self.tracker.metrics)
        self.assertIn("loss", self.tracker.metrics)
        self.assertEqual(len(self.tracker.metrics["accuracy"]), 1)
        self.assertEqual(len(self.tracker.metrics["loss"]), 1)

    def test_metrics_tracker_update_multiple_times(self):
        """测试多次更新指标"""
        self.tracker.update({"accuracy": 0.8}, round_num=1)
        self.tracker.update({"accuracy": 0.85}, round_num=2)
        self.tracker.update({"accuracy": 0.9}, round_num=3)

        self.assertEqual(len(self.tracker.metrics["accuracy"]), 3)
        self.assertEqual(len(self.tracker.history), 3)

    def test_metrics_tracker_get_latest(self):
        """测试获取最新指标值"""
        self.tracker.update({"accuracy": 0.8}, round_num=1)
        self.tracker.update({"accuracy": 0.9}, round_num=2)

        latest = self.tracker.get_latest("accuracy")
        self.assertEqual(latest, 0.9)

    def test_metrics_tracker_get_latest_nonexistent(self):
        """测试获取不存在的指标返回0"""
        latest = self.tracker.get_latest("nonexistent")
        self.assertEqual(latest, 0.0)

    def test_metrics_tracker_get_average(self):
        """测试获取平均指标值"""
        self.tracker.update({"accuracy": 0.8}, round_num=1)
        self.tracker.update({"accuracy": 0.9}, round_num=2)
        self.tracker.update({"accuracy": 1.0}, round_num=3)

        avg = self.tracker.get_average("accuracy")
        self.assertAlmostEqual(avg, 0.9, places=5)

    def test_metrics_tracker_to_dict(self):
        """测试转换为字典"""
        self.tracker.update({"accuracy": 0.9, "loss": 0.1}, round_num=1)
        result = self.tracker.to_dict()

        self.assertIn("latest", result)
        self.assertIn("average", result)
        self.assertIn("history", result)
        self.assertEqual(result["latest"]["accuracy"], 0.9)

    def test_metrics_tracker_save(self):
        """测试保存到文件"""
        self.tracker.update({"accuracy": 0.9}, round_num=1)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            self.tracker.save(temp_file)
            self.assertTrue(os.path.exists(temp_file))

            # 验证文件内容
            import json

            with open(temp_file, "r") as f:
                data = json.load(f)
                self.assertIn("latest", data)
                self.assertEqual(data["latest"]["accuracy"], 0.9)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestComputeAccuracy(unittest.TestCase):
    """测试compute_accuracy函数"""

    def setUp(self):
        """创建简单的模型和数据"""
        self.model = nn.Linear(10, 2)
        self.device = "cpu"

    def test_compute_accuracy_with_correct_predictions(self):
        """测试计算准确率（全正确）"""
        # 创建数据使得所有预测都正确
        data = torch.randn(10, 10)
        targets = torch.zeros(10, dtype=torch.long)

        # 设置模型权重使预测为0
        with torch.no_grad():
            self.model.weight.fill_(0)
            self.model.bias.fill_(-10)  # 使预测为0

        dataset = torch.utils.data.TensorDataset(data, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=5)

        accuracy = compute_accuracy(self.model, loader, self.device)
        self.assertEqual(accuracy, 1.0)

    def test_compute_accuracy_with_half_correct(self):
        """测试计算准确率（一半正确）"""
        # 使用固定的数据，使结果可预测
        data = torch.zeros(10, 10)
        targets = torch.cat(
            [torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long)]
        )

        # 创建简单的模型：输入0，输出仅由偏置决定
        model = nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.fill_(0)
            model.bias[0] = -10  # 输出0的逻辑为负，预测为类别1
            model.bias[1] = 10  # 输出1的逻辑为正，预测为类别0

        dataset = torch.utils.data.TensorDataset(data, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        # 所有输入都一样，所以模型预测应该一致
        accuracy = compute_accuracy(model, loader, self.device)
        # 由于所有输入相同，准确率应该是0.5（一半正确，一半错误）
        self.assertEqual(accuracy, 0.5)

    def test_compute_accuracy_with_empty_data(self):
        """测试空数据集返回0"""
        data = torch.randn(0, 10)
        targets = torch.zeros(0, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(data, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=5)

        accuracy = compute_accuracy(self.model, loader, self.device)
        self.assertEqual(accuracy, 0.0)


class TestComputeLoss(unittest.TestCase):
    """测试compute_loss函数"""

    def setUp(self):
        """创建简单的模型和数据"""
        self.model = nn.Linear(10, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.device = "cpu"

    def test_compute_loss_returns_float(self):
        """测试compute_loss返回浮点数"""
        data = torch.randn(10, 10)
        targets = torch.zeros(10, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(data, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=5)

        loss = compute_loss(self.model, loader, self.criterion, self.device)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)


class TestComputeFingerprintSimilarity(unittest.TestCase):
    """测试compute_fingerprint_similarity函数"""

    def test_compute_fingerprint_similarity_identical(self):
        """测试计算相同指纹的相似度"""
        fp1 = torch.randn(128)
        fp2 = fp1.clone()

        similarity = compute_fingerprint_similarity(fp1, fp2)
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_compute_fingerprint_similarity_orthogonal(self):
        """测试计算正交指纹的相似度"""
        fp1 = torch.randn(128)
        fp2 = torch.randn(128)
        # 使它们正交
        fp2 = fp2 - torch.dot(fp1, fp2) * fp1 / torch.dot(fp1, fp1)
        fp2 = fp2 / (fp2.norm() + 1e-8)

        similarity = compute_fingerprint_similarity(fp1, fp2)
        self.assertAlmostEqual(similarity, 0.0, places=1)

    def test_compute_fingerprint_similarity_opposite(self):
        """测试计算相反指纹的相似度"""
        fp1 = torch.randn(128)
        fp2 = -fp1

        similarity = compute_fingerprint_similarity(fp1, fp2)
        self.assertAlmostEqual(similarity, -1.0, places=5)


class TestComputeVerificationMetrics(unittest.TestCase):
    """测试compute_verification_metrics函数"""

    def test_compute_verification_metrics_all_correct(self):
        """测试计算验证指标（全正确）"""
        predictions = [True, True, True, True, True]
        ground_truth = [True, True, True, True, True]

        metrics = compute_verification_metrics(predictions, ground_truth)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_compute_verification_metrics_half_correct(self):
        """测试计算验证指标（一半正确）"""
        predictions = [True, False, True, False, True]
        ground_truth = [True, True, False, False, True]

        metrics = compute_verification_metrics(predictions, ground_truth)
        self.assertEqual(metrics["accuracy"], 0.6)

    def test_compute_verification_metrics_all_negative(self):
        """测试计算验证指标（全负）"""
        predictions = [False, False, False]
        ground_truth = [False, False, False]

        metrics = compute_verification_metrics(predictions, ground_truth)
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
