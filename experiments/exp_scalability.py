"""可扩展性实验脚本。

本文件属于 FedTracker-Pro 项目
功能: 运行不同客户端规模场景并评估全局模型性能
依赖: src.core, experiments.utils

代码生成来源: implementation_plan.md
章节: 第五阶段 实验系统
生成日期: 2026-04-17
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.utils.data_utils import set_seed

from experiments.utils import (
    build_model_from_config,
    create_experiment_dir,
    save_results,
)


def generate_client_scenarios(
    min_clients: int = 10,
    max_clients: int = 100,
    step: int = 10,
) -> list[int]:
    """生成可扩展性实验的客户端数量场景。"""
    if min_clients <= 0:
        raise ValueError("min_clients must be greater than 0")
    if max_clients < min_clients:
        raise ValueError("max_clients must be >= min_clients")
    if step <= 0:
        raise ValueError("step must be greater than 0")

    scenarios = list(range(min_clients, max_clients + 1, step))
    if scenarios[-1] != max_clients:
        scenarios.append(max_clients)
    return scenarios


def run_scalability_experiment(
    config_path: str,
    min_clients: int = 10,
    max_clients: int = 100,
    step: int = 10,
    num_rounds: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """运行可扩展性实验并返回结果。"""
    base_config = Config(config_path)
    set_seed(base_config.system.seed)

    scenarios = generate_client_scenarios(
        min_clients=min_clients,
        max_clients=max_clients,
        step=step,
    )
    all_results: Dict[str, Dict[str, float]] = {}

    for num_clients in scenarios:
        config = copy.deepcopy(base_config)
        config.federated.num_clients = num_clients

        model = build_model_from_config(config)
        framework = FedTrackerPro(config)
        framework.initialize(model)
        framework.train(num_rounds=num_rounds)

        if framework.server is None or framework.data_manager is None:
            raise RuntimeError("Framework is not initialized")

        metrics = framework.server.evaluate(framework.data_manager.get_test_loader())
        all_results[f"clients_{num_clients}"] = {
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "loss": float(metrics.get("loss", float("inf"))),
        }

    base_dir = output_dir if output_dir is not None else "./experiments/results"
    exp_dir = create_experiment_dir(base_dir=base_dir)
    result_path = save_results(all_results, exp_dir, "scalability_results.json")

    return {
        "results": all_results,
        "experiment_dir": exp_dir,
        "results_path": result_path,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 FedTracker-Pro 可扩展性实验")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--min-clients", type=int, default=10, help="最小客户端数量")
    parser.add_argument("--max-clients", type=int, default=100, help="最大客户端数量")
    parser.add_argument("--step", type=int, default=10, help="客户端数量步长")
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="覆盖配置中的全局训练轮数",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="实验结果输出根目录（默认: ./experiments/results）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)
    payload = run_scalability_experiment(
        config_path=args.config,
        min_clients=args.min_clients,
        max_clients=args.max_clients,
        step=args.step,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
    )
    print(f"[scalability] results saved to: {payload['results_path']}")
    print(payload["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
