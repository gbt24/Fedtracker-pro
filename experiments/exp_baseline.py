"""基线对比实验脚本。

本文件属于 FedTracker-Pro 项目
功能: 执行基线实验（训练 + 3 种攻击鲁棒性评估）
依赖: src.core, src.models, src.attacks, experiments.utils

代码生成来源: implementation_plan.md
章节: 第五阶段 实验系统
生成日期: 2026-04-17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import FineTuningAttack, PruningAttack, QuantizationAttack
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.utils.data_utils import set_seed

from experiments.utils import (
    build_model_from_config,
    create_experiment_dir,
    resolve_progress_flag,
    save_results,
)


def build_default_attacks(device: str = "cuda") -> List:
    """构建基线实验默认攻击列表。"""
    return [
        FineTuningAttack(device=device),
        PruningAttack(device=device),
        QuantizationAttack(device=device),
    ]


def run_baseline_experiment(
    config_path: str,
    num_rounds: Optional[int] = None,
    output_dir: Optional[str] = None,
    show_progress: bool = False,
    enforce_crypto: bool = True,
    enforce_watermark: bool = False,
) -> Dict[str, Any]:
    """运行基线实验并返回结果。"""
    config = Config(config_path)
    set_seed(config.system.seed)

    model = build_model_from_config(config)
    framework = FedTrackerPro(config)
    framework.initialize(model)
    framework.train(
        num_rounds=num_rounds,
        show_progress=show_progress,
        progress_desc="baseline-train",
    )

    if framework.data_manager is None:
        raise RuntimeError("Data manager is not initialized")

    attacks = build_default_attacks(device=framework.device)
    test_loader = framework.data_manager.get_test_loader()
    results = framework.evaluate_attack_robustness(
        attacks,
        test_loader,
        show_progress=show_progress,
        progress_desc="baseline-attacks",
        enforce_crypto=enforce_crypto,
        enforce_watermark=enforce_watermark,
    )

    base_dir = output_dir if output_dir is not None else "./experiments/results"
    exp_dir = create_experiment_dir(base_dir=base_dir)
    result_path = save_results(results, exp_dir, "baseline_results.json")

    return {
        "results": results,
        "experiment_dir": exp_dir,
        "results_path": result_path,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 FedTracker-Pro 基线实验")
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
    crypto_group = parser.add_mutually_exclusive_group()
    crypto_group.add_argument(
        "--enforce-crypto",
        dest="enforce_crypto",
        action="store_true",
        help="攻击评估时强制密码学校验（默认）",
    )
    crypto_group.add_argument(
        "--relax-crypto-check",
        dest="enforce_crypto",
        action="store_false",
        help="攻击评估时不把密码学校验作为硬性拒绝条件",
    )
    watermark_group = parser.add_mutually_exclusive_group()
    watermark_group.add_argument(
        "--enforce-watermark",
        dest="enforce_watermark",
        action="store_true",
        help="攻击评估时强制水印校验",
    )
    watermark_group.add_argument(
        "--relax-watermark-check",
        dest="enforce_watermark",
        action="store_false",
        help="攻击评估时不把水印校验作为硬性拒绝条件（默认）",
    )
    parser.set_defaults(progress=None)
    parser.set_defaults(enforce_crypto=True)
    parser.set_defaults(enforce_watermark=False)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)
    payload = run_baseline_experiment(
        config_path=args.config,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
        show_progress=resolve_progress_flag(args.progress),
        enforce_crypto=args.enforce_crypto,
        enforce_watermark=args.enforce_watermark,
    )
    print(f"[baseline] results saved to: {payload['results_path']}")
    print(payload["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
