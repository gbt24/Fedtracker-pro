"""密码学验证模块。

本文件属于 FedTracker-Pro 项目
功能: 对模型进行签名嵌入与完整性验证
依赖: torch, src.utils.crypto_utils

代码生成来源: implementation_plan.md
章节: 阶段5 密码学验证系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..utils.crypto_utils import CryptoManager


class CryptographicVerification:
    """使用 ECDSA 进行模型所有权验证。"""

    def __init__(
        self,
        key_size: int = 256,
        scheme: str = "ecdsa",
        hash_algorithm: str = "sha256",
        strength: float = 0.01,
        device: str = "cuda",
    ) -> None:
        _ = hash_algorithm
        if strength <= 0:
            raise ValueError("strength must be greater than 0")
        self.crypto_manager = CryptoManager(key_size=key_size, scheme=scheme)
        self.crypto_manager.generate_keypair()
        self.strength = strength
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.signature_num_bits: int | None = None
        self.expected_signature: bytes | None = None
        self.expected_bits: torch.Tensor | None = None
        self.signed_message: Dict[str, Any] | None = None
        self.embedded_model_hash: str | None = None

    def _build_message(self, model: nn.Module) -> Dict[str, Any]:
        """根据模型状态构造待签名消息。"""
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return {"model_hash": self.crypto_manager.hash_model_state(state)}

    def _embed_signature_bits(
        self, model: nn.Module, signature: bytes, num_bits: int
    ) -> nn.Module:
        """按指定比特数将签名嵌入模型。"""
        sig_bits = self.crypto_manager.encode_signature_to_bits(
            signature, num_bits=num_bits
        )
        embeddable_params = [
            param
            for name, param in model.named_parameters()
            if "weight" in name and param.requires_grad
        ]
        bit_idx = 0
        with torch.no_grad():
            for param in embeddable_params:
                if bit_idx >= len(sig_bits):
                    break
                flat_param = param.view(-1)
                num_local_bits = min(len(flat_param), len(sig_bits) - bit_idx)
                for i in range(num_local_bits):
                    val = flat_param[i].item()
                    bit = sig_bits[bit_idx + i].item()
                    quantized = int(round(val / self.strength))
                    if bit > 0.5:
                        new_val = (quantized | 1) * self.strength
                    else:
                        new_val = (quantized & ~1) * self.strength
                    flat_param[i] = new_val
                bit_idx += num_local_bits
        return model

    def _extract_signature_bits(self, model: nn.Module, num_bits: int) -> torch.Tensor:
        """按指定比特数从模型提取签名比特。"""
        bits: list[float] = []
        embeddable_params = [
            param
            for name, param in model.named_parameters()
            if "weight" in name and param.requires_grad
        ]
        for param in embeddable_params:
            if len(bits) >= num_bits:
                break
            flat_param = param.view(-1)
            remaining = num_bits - len(bits)
            for i in range(min(len(flat_param), remaining)):
                val = flat_param[i].item()
                bits.append(float(int(round(val / self.strength)) & 1))
        return torch.tensor(bits, dtype=torch.float32)

    def embed_to_model(self, model: nn.Module) -> nn.Module:
        """生成签名并嵌入模型参数。"""
        message = self._build_message(model)
        signature = self.crypto_manager.sign(message)
        self.signed_message = message
        capacity = sum(
            param.numel()
            for name, param in model.named_parameters()
            if "weight" in name and param.requires_grad
        )
        self.signature_num_bits = min(256, capacity)
        signature_bytes = signature[: max(1, (self.signature_num_bits + 7) // 8)]
        self.expected_bits = self.crypto_manager.encode_signature_to_bits(
            signature_bytes,
            num_bits=self.signature_num_bits,
        )
        self.expected_signature = signature_bytes
        signed_model = self._embed_signature_bits(
            model, signature_bytes, self.signature_num_bits
        )
        self.embedded_model_hash = self._build_message(signed_model)["model_hash"]
        return signed_model

    def verify_model(self, model: nn.Module) -> Dict[str, Any]:
        """验证模型中嵌入签名与当前模型状态是否匹配。"""
        if (
            self.expected_signature is None
            or self.expected_bits is None
            or self.signed_message is None
            or self.signature_num_bits is None
        ):
            raise ValueError("No embedded signature available for verification")
        current_message = self._build_message(model)
        extracted_bits = self._extract_signature_bits(model, self.signature_num_bits)
        signature = self.crypto_manager.decode_bits_to_signature(extracted_bits)
        is_valid = (
            bool(torch.equal(extracted_bits, self.expected_bits))
            and current_message["model_hash"] == self.embedded_model_hash
        )
        return {
            "is_valid": is_valid,
            "signature": signature,
            "message": current_message,
        }
