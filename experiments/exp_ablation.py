"""消融实验配置脚本。

本文件属于 FedTracker-Pro 项目
功能: 提供消融实验分组定义
依赖: 无

代码生成来源: implementation_plan.md
章节: 第五阶段 实验系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Dict


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
