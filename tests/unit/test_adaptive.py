"""自适应分配模块单元测试。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defense.adaptive_allocation import AdaptiveAllocator


class TestAdaptiveAllocator(unittest.TestCase):
    """测试自适应分配器行为。"""

    def test_evaluate_tolerance_prefers_high_accuracy_and_similarity(self) -> None:
        allocator = AdaptiveAllocator(beta=0.2, min_allocation=0.05)
        score = allocator.evaluate_tolerance(
            accuracy=0.9,
            loss=0.2,
            fingerprint_similarity=0.85,
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_allocate_respects_minimum_share(self) -> None:
        allocator = AdaptiveAllocator(beta=0.3, min_allocation=0.1)
        allocations = allocator.allocate({"c1": 0.8, "c2": 0.2})
        self.assertAlmostEqual(sum(allocations.values()), 0.3, places=6)
        self.assertGreaterEqual(allocations["c1"], 0.1)
        self.assertGreaterEqual(allocations["c2"], 0.1)


if __name__ == "__main__":
    unittest.main()
