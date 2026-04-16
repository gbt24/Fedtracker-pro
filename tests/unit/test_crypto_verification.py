"""密码学验证模块单元测试。"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defense.crypto_verification import CryptographicVerification


class TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(12, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestCryptographicVerification(unittest.TestCase):
    """测试模型签名嵌入与验证。"""

    def test_embed_and_verify_model(self) -> None:
        verifier = CryptographicVerification(device="cpu")
        model = TinyLinear()
        signed = verifier.embed_to_model(model)
        result = verifier.verify_model(signed)
        self.assertTrue(result["is_valid"])
        self.assertIn("signature", result)

    def test_verify_fails_when_model_changes(self) -> None:
        verifier = CryptographicVerification(device="cpu")
        model = TinyLinear()
        signed = verifier.embed_to_model(model)
        with torch.no_grad():
            signed.fc.weight.add_(0.5)
        result = verifier.verify_model(signed)
        self.assertFalse(result["is_valid"])

    def test_verify_handles_non_byte_aligned_capacity(self) -> None:
        class TinyOdd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(13, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        verifier = CryptographicVerification(device="cpu")
        model = TinyOdd()
        signed = verifier.embed_to_model(model)
        result = verifier.verify_model(signed)
        self.assertTrue(result["is_valid"])

    def test_verify_raises_when_never_embedded(self) -> None:
        verifier = CryptographicVerification(device="cpu")
        model = TinyLinear()
        with self.assertRaises(ValueError):
            verifier.verify_model(model)

    def test_invalid_strength_raises(self) -> None:
        with self.assertRaises(ValueError):
            CryptographicVerification(device="cpu", strength=0.0)


if __name__ == "__main__":
    unittest.main()
