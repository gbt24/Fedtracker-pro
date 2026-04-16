"""三层验证集成模块。

本文件属于 FedTracker-Pro 项目
功能: 聚合统计、密码学和行为层验证结果
依赖: typing

代码生成来源: implementation_plan.md
章节: 阶段5 三层验证集成
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Any, Dict


class MultiLayerVerifier:
    """三层所有权验证器。"""

    def __init__(
        self,
        watermark,
        fingerprint,
        level1_threshold: float = 0.75,
        level3_threshold: float = 0.9,
    ) -> None:
        self.watermark = watermark
        self.fingerprint = fingerprint
        self.level1_threshold = level1_threshold
        self.level3_threshold = level3_threshold

    def verify_ownership(
        self, model, crypto_result: bool | Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行三层验证并返回综合结果。"""
        level1_score = float(self.fingerprint.verify(model))
        level3_score = float(self.watermark.verify(model))

        level1 = {
            "passed": level1_score >= self.level1_threshold,
            "score": level1_score,
        }
        if isinstance(crypto_result, dict):
            level2_passed = bool(crypto_result.get("is_valid", False))
        else:
            level2_passed = bool(crypto_result)
        level2 = {"passed": level2_passed}
        level3 = {
            "passed": level3_score >= self.level3_threshold,
            "score": level3_score,
        }
        is_verified = level1["passed"] and level2["passed"] and level3["passed"]

        return {
            "is_verified": is_verified,
            "level1": level1,
            "level2": level2,
            "level3": level3,
        }
