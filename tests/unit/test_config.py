"""Config 模块单元测试。"""

import os
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import src.core.config as config_module
from src.core.config import Config, get_config, set_config


class TestConfig(unittest.TestCase):
    """测试配置系统行为。"""

    def tearDown(self) -> None:
        config_module._global_config = None

    def test_config_has_default_sections(self) -> None:
        cfg = Config()
        self.assertEqual(cfg.federated.num_clients, 50)
        self.assertEqual(cfg.data.dataset, "cifar10")

    def test_config_can_save_and_load_yaml(self) -> None:
        cfg = Config()
        cfg.federated.num_clients = 12
        cfg.data.alpha = 0.3

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "config.yaml")
            cfg.save_to_yaml(file_path)

            loaded = Config(file_path)
            self.assertEqual(loaded.federated.num_clients, 12)
            self.assertAlmostEqual(loaded.data.alpha, 0.3)

    def test_config_load_supports_adaptive_allocation_key(self) -> None:
        data = {
            "adaptive_allocation": {
                "enabled": True,
                "evaluation_period": 5,
                "beta": 0.2,
                "min_allocation": 0.1,
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "adaptive.yaml")
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)

            loaded = Config(file_path)
            self.assertEqual(loaded.adaptive.evaluation_period, 5)
            self.assertAlmostEqual(loaded.adaptive.beta, 0.2)

    def test_get_and_set_global_config(self) -> None:
        cfg1 = get_config()
        cfg2 = get_config()
        self.assertIs(cfg1, cfg2)

        fresh = Config()
        fresh.federated.num_clients = 99
        set_config(fresh)

        cfg3 = get_config()
        self.assertIs(cfg3, fresh)
        self.assertEqual(cfg3.federated.num_clients, 99)

    def test_init_raises_when_config_path_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Config("/tmp/does-not-exist-fedtracker-config.yaml")


if __name__ == "__main__":
    unittest.main()
