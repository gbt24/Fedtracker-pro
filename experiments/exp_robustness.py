"""鲁棒性实验脚本。

本文件属于 FedTracker-Pro 项目
功能: 执行鲁棒性实验（训练 + 6 种攻击评估）
依赖: src.core, src.attacks, experiments.utils

代码生成来源: code_generation_guide.md
章节: 阶段8 实验脚本
生成日期: 2026-04-17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import (
    AmbiguityAttack,
    FineTuningAttack,
    ModelExtractionAttack,
    OverwritingAttack,
    PruningAttack,
    QuantizationAttack,
)
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.utils.data_utils import set_seed

from experiments.utils import (
    build_model_from_config,
    create_experiment_dir,
    resolve_progress_flag,
    save_results,
)


def build_robustness_attacks(device: str = "cuda") -> List:
    """构建鲁棒性实验攻击列表。"""
    return [
        FineTuningAttack(device=device),
        PruningAttack(device=device),
        QuantizationAttack(device=device),
        OverwritingAttack(device=device),
        AmbiguityAttack(device=device),
        ModelExtractionAttack(device=device),
    ]


def run_robustness_experiment(
    config_path: str,
    num_rounds: Optional[int] = None,
    output_dir: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """运行鲁棒性实验并返回结果。"""
    config = Config(config_path)
    set_seed(config.system.seed)

    model = build_model_from_config(config)
    framework = FedTrackerPro(config)
    framework.initialize(model)
    framework.train(
        num_rounds=num_rounds,
        show_progress=show_progress,
        progress_desc="robustness-train",
    )

    if framework.data_manager is None:
        raise RuntimeError("Data manager is not initialized")

    attacks = build_robustness_attacks(device=framework.device)
    test_loader = framework.data_manager.get_test_loader()
    results = framework.evaluate_attack_robustness(
        attacks,
        test_loader,
        show_progress=show_progress,
        progress_desc="robustness-attacks",
    )

    base_dir = output_dir if output_dir is not None else "./experiments/results"
    exp_dir = create_experiment_dir(base_dir=base_dir)
    result_path = save_results(results, exp_dir, "robustness_results.json")

    return {
        "results": results,
        "experiment_dir": exp_dir,
        "results_path": result_path,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 FedTracker-Pro 鲁棒性实验")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
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
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="显示进度条",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="关闭进度条",
    )
    parser.set_defaults(progress=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)
    payload = run_robustness_experiment(
        config_path=args.config,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
        show_progress=resolve_progress_flag(args.progress),
    )
    print(f"[robustness] results saved to: {payload['results_path']}")
    print(payload["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
