"""实验结果读取脚本。

本文件属于 FedTracker-Pro 项目
功能: 读取并汇总 experiments/results 目录下的实验结果 JSON
依赖: argparse, json, os
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


RESULT_FILENAMES = {
    "baseline": "baseline_results.json",
    "robustness": "robustness_results.json",
    "ablation": "ablation_results.json",
    "scalability": "scalability_results.json",
}


def _is_timestamp_dir(name: str) -> bool:
    """判断目录名是否符合 YYYYMMDD_HHMMSS。"""
    if len(name) != 15 or name[8] != "_":
        return False
    return name[:8].isdigit() and name[9:].isdigit()


def list_experiment_dirs(root: str) -> List[str]:
    """列出 root 下时间戳实验目录，按新到旧排序。"""
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []

    candidates = [
        str(path)
        for path in root_path.iterdir()
        if path.is_dir() and _is_timestamp_dir(path.name)
    ]
    return sorted(candidates, reverse=True)


def resolve_target_dir(root: str, latest: bool, specified_dir: Optional[str]) -> str:
    """解析目标实验目录。"""
    if specified_dir:
        target = os.path.abspath(specified_dir)
        if not os.path.exists(target):
            raise FileNotFoundError(f"Specified directory does not exist: {target}")
        if not os.path.isdir(target):
            raise NotADirectoryError(f"Specified path is not a directory: {target}")
        return target

    if not latest:
        raise ValueError("Either --latest or --dir must be provided")

    exp_dirs = list_experiment_dirs(root)
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found under: {root}")
    return exp_dirs[0]


def collect_result_files(exp_dir: str) -> Dict[str, str]:
    """收集目标实验目录中的结果文件路径。"""
    out: Dict[str, str] = {}
    for key, filename in RESULT_FILENAMES.items():
        path = os.path.join(exp_dir, filename)
        if os.path.isfile(path):
            out[key] = path
    return out


def load_results(exp_dir: str) -> Dict[str, Any]:
    """加载目标实验目录中的所有结果。"""
    files = collect_result_files(exp_dir)
    payload: Dict[str, Any] = {
        "experiment_dir": os.path.abspath(exp_dir),
        "results": {},
    }
    for key, path in files.items():
        with open(path, "r", encoding="utf-8") as f:
            payload["results"][key] = json.load(f)
    return payload


def _format_key_values(data: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key in sorted(data.keys()):
        lines.append(f"  - {key}: {data[key]}")
    return lines


def format_results_table(payload: Dict[str, Any]) -> str:
    """将结果格式化为文本简表。"""
    lines: List[str] = []
    lines.append(f"Experiment Dir: {payload['experiment_dir']}")
    results: Dict[str, Any] = payload.get("results", {})

    if not results:
        lines.append("No result json files found.")
        return "\n".join(lines)

    for exp_name in ["baseline", "robustness", "ablation", "scalability"]:
        if exp_name not in results:
            continue
        lines.append(f"\n[{exp_name}]")
        block = results[exp_name]

        if exp_name in {"baseline", "robustness"}:
            if isinstance(block, dict):
                lines.extend(_format_key_values(block))
            else:
                lines.append(f"  - value: {block}")
            continue

        if exp_name == "ablation":
            for group_name in sorted(block.keys()):
                lines.append(f"  * {group_name}")
                metrics = block[group_name]
                if isinstance(metrics, dict):
                    for metric_name in sorted(metrics.keys()):
                        lines.append(f"    - {metric_name}: {metrics[metric_name]}")
                else:
                    lines.append(f"    - value: {metrics}")
            continue

        if exp_name == "scalability":
            for scenario in sorted(block.keys()):
                lines.append(f"  * {scenario}")
                metrics = block[scenario]
                if isinstance(metrics, dict):
                    for metric_name in sorted(metrics.keys()):
                        lines.append(f"    - {metric_name}: {metrics[metric_name]}")
                else:
                    lines.append(f"    - value: {metrics}")
            continue

    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="读取并汇总实验结果")
    parser.add_argument(
        "--root",
        type=str,
        default="./experiments/results",
        help="实验结果根目录",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="读取最新时间戳目录（默认行为）",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="指定实验目录（优先于 --latest）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        help="输出格式",
    )
    parser.set_defaults(latest=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)
    try:
        target_dir = resolve_target_dir(
            root=args.root,
            latest=bool(args.latest),
            specified_dir=args.dir,
        )
        payload = load_results(target_dir)
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        json.JSONDecodeError,
    ) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_results_table(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
