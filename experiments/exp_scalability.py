"""可扩展性实验脚本。

本文件属于 FedTracker-Pro 项目
功能: 生成客户端规模实验场景
依赖: 无

代码生成来源: implementation_plan.md
章节: 第五阶段 实验系统
生成日期: 2026-04-16
"""

from __future__ import annotations


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
