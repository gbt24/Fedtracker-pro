"""消融实验脚本。

本文件属于 FedTracker-Pro 项目
功能: 执行 6 组防御开关的消融实验并评估鲁棒性
依赖: src.core, experiments.exp_robustness, experiments.utils

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

from experiments.exp_robustness import build_robustness_attacks
from experiments.utils import (
    build_model_from_config,
    create_experiment_dir,
    progress_iter,
    resolve_progress_flag,
    save_results,
)


def get_ablation_groups() -> Dict[str, Dict[str, bool]]:
    """返回消融实验组配置。"""
    return {
        "baseline": {
            "watermark": False,
            "fingerprint": False,
            "adaptive": False,
            "crypto": False,
            "unlearning": False,
        },
        "watermark_only": {
            "watermark": True,
            "fingerprint": False,
            "adaptive": False,
            "crypto": False,
            "unlearning": False,
        },
        "fingerprint_only": {
            "watermark": False,
            "fingerprint": True,
            "adaptive": False,
            "crypto": False,
            "unlearning": False,
        },
        "adaptive": {
            "watermark": True,
            "fingerprint": True,
            "adaptive": True,
            "crypto": False,
            "unlearning": False,
        },
        "crypto": {
            "watermark": True,
            "fingerprint": True,
            "adaptive": True,
            "crypto": True,
            "unlearning": False,
        },
        "full": {
            "watermark": True,
            "fingerprint": True,
            "adaptive": True,
            "crypto": True,
            "unlearning": True,
        },
    }


def _apply_group_flags(config: Config, flags: Dict[str, bool]) -> None:
    """将消融组开关写入配置。"""
    config.watermark.enabled = flags["watermark"]
    config.fingerprint.enabled = flags["fingerprint"]
    config.adaptive.enabled = flags["adaptive"]
    config.crypto.enabled = flags["crypto"]
    config.unlearning.enabled = flags["unlearning"]


def run_ablation_experiment(
    config_path: str,
    num_rounds: Optional[int] = None,
    output_dir: Optional[str] = None,
    show_progress: bool = False,
    enforce_crypto: bool = True,
    enforce_watermark: bool = False,
) -> Dict[str, Any]:
    """运行消融实验并返回结果。"""
    base_config = Config(config_path)
    set_seed(base_config.system.seed)

    all_results: Dict[str, Dict[str, float]] = {}
    groups = get_ablation_groups()
    group_iter = progress_iter(
        groups.items(),
        enabled=show_progress,
        total=len(groups),
        desc="ablation-groups",
        unit="group",
    )

    for group_name, flags in group_iter:
        config = copy.deepcopy(base_config)
        _apply_group_flags(config, flags)

        model = build_model_from_config(config)
        framework = FedTrackerPro(config)
        framework.initialize(model)
        framework.train(
            num_rounds=num_rounds,
            show_progress=show_progress,
            progress_desc=f"ablation-train:{group_name}",
        )

        if framework.data_manager is None:
            raise RuntimeError("Data manager is not initialized")

        attacks = build_robustness_attacks(device=framework.device)
        test_loader = framework.data_manager.get_test_loader()
        all_results[group_name] = framework.evaluate_attack_robustness(
            attacks,
            test_loader,
            show_progress=show_progress,
            progress_desc=f"ablation-attacks:{group_name}",
            enforce_crypto=enforce_crypto,
            enforce_watermark=enforce_watermark,
        )

    base_dir = output_dir if output_dir is not None else "./experiments/results"
    exp_dir = create_experiment_dir(base_dir=base_dir)
    result_path = save_results(all_results, exp_dir, "ablation_results.json")

    return {
        "results": all_results,
        "experiment_dir": exp_dir,
        "results_path": result_path,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 FedTracker-Pro 消融实验")
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
    payload = run_ablation_experiment(
        config_path=args.config,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
        show_progress=resolve_progress_flag(args.progress),
        enforce_crypto=args.enforce_crypto,
        enforce_watermark=args.enforce_watermark,
    )
    print(f"[ablation] results saved to: {payload['results_path']}")
    print(payload["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
