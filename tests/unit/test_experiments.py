"""实验脚本与工具模块测试。"""

import json
import os
import sys
import tempfile
import unittest

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.exp_ablation import get_ablation_groups
from experiments.exp_baseline import build_default_attacks
from experiments.exp_robustness import build_robustness_attacks
from experiments.exp_scalability import generate_client_scenarios
from experiments.utils import (
    aggregate_client_metrics,
    create_experiment_dir,
    save_results,
)


class TestExperimentUtils(unittest.TestCase):
    def test_create_experiment_dir_creates_timestamp_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = create_experiment_dir(base_dir=tmp_dir)
            self.assertTrue(os.path.isdir(exp_dir))
            self.assertRegex(
                os.path.basename(exp_dir),
                r"^\d{8}_\d{6}$",
            )

    def test_save_results_serializes_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = {
                "acc": torch.tensor(0.87),
                "nested": {"loss": torch.tensor(1.23)},
                "scores": [torch.tensor(1.0), torch.tensor(2.0)],
            }
            save_results(payload, save_dir=tmp_dir, filename="result.json")

            with open(os.path.join(tmp_dir, "result.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertAlmostEqual(data["acc"], 0.87, places=6)
            self.assertAlmostEqual(data["nested"]["loss"], 1.23, places=6)
            self.assertEqual(data["scores"], [1.0, 2.0])

    def test_aggregate_client_metrics_returns_stats(self) -> None:
        metrics = [
            {"accuracy": 80.0, "loss": 1.0},
            {"accuracy": 90.0, "loss": 0.5},
            {"accuracy": 85.0, "loss": 0.75},
        ]
        agg = aggregate_client_metrics(metrics)

        self.assertEqual(agg["accuracy_min"], 80.0)
        self.assertEqual(agg["accuracy_max"], 90.0)
        self.assertAlmostEqual(agg["accuracy_mean"], 85.0, places=6)
        self.assertEqual(agg["loss_min"], 0.5)
        self.assertEqual(agg["loss_max"], 1.0)


class TestExperimentScripts(unittest.TestCase):
    def test_build_default_attacks_contains_core_attacks(self) -> None:
        attacks = build_default_attacks(device="cpu")
        names = {attack.get_attack_name() for attack in attacks}
        self.assertTrue({"fine_tuning", "pruning", "quantization"}.issubset(names))

    def test_get_ablation_groups_has_required_groups(self) -> None:
        groups = get_ablation_groups()
        self.assertTrue(
            {
                "baseline",
                "watermark_only",
                "fingerprint_only",
                "adaptive",
                "crypto",
                "full",
            }.issubset(set(groups.keys()))
        )

    def test_build_robustness_attacks_contains_extended_attacks(self) -> None:
        attacks = build_robustness_attacks(device="cpu")
        names = {attack.get_attack_name() for attack in attacks}
        self.assertTrue(
            {
                "fine_tuning",
                "pruning",
                "quantization",
                "overwriting",
                "ambiguity",
                "model_extraction",
            }.issubset(names)
        )

    def test_generate_client_scenarios_monotonic(self) -> None:
        scenarios = generate_client_scenarios(
            min_clients=10,
            max_clients=100,
            step=30,
        )
        self.assertEqual(scenarios, [10, 40, 70, 100])

    def test_generate_client_scenarios_rejects_invalid_args(self) -> None:
        with self.assertRaises(ValueError):
            generate_client_scenarios(min_clients=0, max_clients=10, step=5)
        with self.assertRaises(ValueError):
            generate_client_scenarios(min_clients=10, max_clients=5, step=5)
        with self.assertRaises(ValueError):
            generate_client_scenarios(min_clients=10, max_clients=20, step=0)

    def test_default_config_yaml_exists_and_has_sections(self) -> None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../..",
            "configs",
            "default.yaml",
        )
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for key in [
            "federated",
            "data",
            "model",
            "watermark",
            "fingerprint",
            "adaptive_allocation",
            "crypto",
            "unlearning",
            "verification",
            "system",
        ]:
            self.assertIn(key, cfg)


if __name__ == "__main__":
    unittest.main()
