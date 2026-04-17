"""实验脚本与工具模块测试。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
    resolve_progress_flag,
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

    def test_resolve_progress_flag_prefers_explicit_value(self) -> None:
        self.assertTrue(resolve_progress_flag(True))
        self.assertFalse(resolve_progress_flag(False))

    def test_resolve_progress_flag_follows_tty_when_unspecified(self) -> None:
        with patch("experiments.utils.sys.stderr.isatty", return_value=True):
            self.assertTrue(resolve_progress_flag(None))
        with patch("experiments.utils.sys.stderr.isatty", return_value=False):
            self.assertFalse(resolve_progress_flag(None))


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


class TestExperimentCliEntry(unittest.TestCase):
    def test_baseline_main_delegates_to_runner(self) -> None:
        from experiments import exp_baseline

        with (
            patch.object(
                exp_baseline,
                "run_baseline_experiment",
                return_value={
                    "results": {"fine_tuning": 1.0},
                    "results_path": "a.json",
                },
            ) as mock_run,
            patch("builtins.print") as mock_print,
        ):
            exit_code = exp_baseline.main(
                [
                    "--config",
                    "configs/default.yaml",
                    "--num-rounds",
                    "1",
                    "--no-progress",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once_with(
            config_path="configs/default.yaml",
            num_rounds=1,
            output_dir=None,
            show_progress=False,
        )
        mock_print.assert_called()

    def test_ablation_main_delegates_to_runner(self) -> None:
        from experiments import exp_ablation

        with (
            patch.object(
                exp_ablation,
                "run_ablation_experiment",
                return_value={
                    "results": {"full": {"fine_tuning": 1.0}},
                    "results_path": "b.json",
                },
            ) as mock_run,
            patch("builtins.print") as mock_print,
        ):
            exit_code = exp_ablation.main(
                [
                    "--config",
                    "configs/default.yaml",
                    "--num-rounds",
                    "1",
                    "--no-progress",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once_with(
            config_path="configs/default.yaml",
            num_rounds=1,
            output_dir=None,
            show_progress=False,
        )
        mock_print.assert_called()

    def test_robustness_main_delegates_to_runner(self) -> None:
        from experiments import exp_robustness

        with (
            patch.object(
                exp_robustness,
                "run_robustness_experiment",
                return_value={"results": {"ambiguity": 1.0}, "results_path": "c.json"},
            ) as mock_run,
            patch("builtins.print") as mock_print,
        ):
            exit_code = exp_robustness.main(
                [
                    "--config",
                    "configs/default.yaml",
                    "--num-rounds",
                    "1",
                    "--no-progress",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once_with(
            config_path="configs/default.yaml",
            num_rounds=1,
            output_dir=None,
            show_progress=False,
        )
        mock_print.assert_called()

    def test_scalability_main_delegates_to_runner(self) -> None:
        from experiments import exp_scalability

        with (
            patch.object(
                exp_scalability,
                "run_scalability_experiment",
                return_value={
                    "results": {"clients_10": {"accuracy": 10.0, "loss": 1.0}},
                    "results_path": "d.json",
                },
            ) as mock_run,
            patch("builtins.print") as mock_print,
        ):
            exit_code = exp_scalability.main(
                [
                    "--config",
                    "configs/default.yaml",
                    "--num-rounds",
                    "1",
                    "--min-clients",
                    "10",
                    "--max-clients",
                    "20",
                    "--step",
                    "10",
                    "--no-progress",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once_with(
            config_path="configs/default.yaml",
            min_clients=10,
            max_clients=20,
            step=10,
            num_rounds=1,
            output_dir=None,
            show_progress=False,
        )
        mock_print.assert_called()

    def test_baseline_main_supports_explicit_progress_flag(self) -> None:
        from experiments import exp_baseline

        with (
            patch.object(
                exp_baseline,
                "run_baseline_experiment",
                return_value={
                    "results": {"fine_tuning": 1.0},
                    "results_path": "a.json",
                },
            ) as mock_run,
            patch("builtins.print"),
        ):
            exit_code = exp_baseline.main(
                ["--config", "configs/default.yaml", "--num-rounds", "1", "--progress"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run.call_args.kwargs["show_progress"], True)

    def test_ablation_main_supports_explicit_progress_flag(self) -> None:
        from experiments import exp_ablation

        with (
            patch.object(
                exp_ablation,
                "run_ablation_experiment",
                return_value={
                    "results": {"full": {"fine_tuning": 1.0}},
                    "results_path": "b.json",
                },
            ) as mock_run,
            patch("builtins.print"),
        ):
            exit_code = exp_ablation.main(
                ["--config", "configs/default.yaml", "--num-rounds", "1", "--progress"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run.call_args.kwargs["show_progress"], True)

    def test_robustness_main_supports_explicit_progress_flag(self) -> None:
        from experiments import exp_robustness

        with (
            patch.object(
                exp_robustness,
                "run_robustness_experiment",
                return_value={"results": {"ambiguity": 1.0}, "results_path": "c.json"},
            ) as mock_run,
            patch("builtins.print"),
        ):
            exit_code = exp_robustness.main(
                ["--config", "configs/default.yaml", "--num-rounds", "1", "--progress"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run.call_args.kwargs["show_progress"], True)

    def test_scalability_main_supports_explicit_progress_flag(self) -> None:
        from experiments import exp_scalability

        with (
            patch.object(
                exp_scalability,
                "run_scalability_experiment",
                return_value={
                    "results": {"clients_10": {"accuracy": 10.0, "loss": 1.0}},
                    "results_path": "d.json",
                },
            ) as mock_run,
            patch("builtins.print"),
        ):
            exit_code = exp_scalability.main(
                [
                    "--config",
                    "configs/default.yaml",
                    "--num-rounds",
                    "1",
                    "--min-clients",
                    "10",
                    "--max-clients",
                    "20",
                    "--step",
                    "10",
                    "--progress",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run.call_args.kwargs["show_progress"], True)

    def test_module_help_has_no_runtime_warning(self) -> None:
        repo_root = os.path.join(os.path.dirname(__file__), "../..")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.exp_baseline",
                "--help",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertNotIn("RuntimeWarning", completed.stderr)


if __name__ == "__main__":
    unittest.main()
