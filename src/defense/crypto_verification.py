"""密码学验证模块。

本文件属于 FedTracker-Pro 项目
功能: 对模型进行签名嵌入与完整性验证
依赖: torch, src.utils.crypto_utils

代码生成来源: implementation_plan.md
章节: 阶段5 密码学验证系统
生成日期: 2026-04-16
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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
        if hash_algorithm.lower() != "sha256":
            raise ValueError("Only sha256 is currently supported")
        if strength <= 0:
            raise ValueError("strength must be greater than 0")
        self.crypto_manager = CryptoManager(key_size=key_size, scheme=scheme)
        self.crypto_manager.generate_keypair()
        self.hash_algorithm = hash_algorithm.lower()
        self.strength = strength
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.signature_num_bits: int | None = None
        self.expected_signature: bytes | None = None
        self.embedded_signature: bytes | None = None
        self.expected_bits: torch.Tensor | None = None
        self.signed_message: Dict[str, Any] | None = None
        self.embedded_model_hash: str | None = None

    def _build_message(
        self, model: nn.Module, client_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """根据模型状态构造待签名消息。"""
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        message: Dict[str, Any] = {
            "model_hash": self.crypto_manager.hash_model_state(state)
        }
        if client_id is not None:
            message["client_id"] = client_id
        return message

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

    def embed_to_model(
        self, model: nn.Module, client_id: Optional[int] = None
    ) -> nn.Module:
        """生成签名并嵌入模型参数。"""
        message = self._build_message(model, client_id=client_id)
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
        self.expected_signature = signature
        self.embedded_signature = signature_bytes
        signed_model = self._embed_signature_bits(
            model, signature_bytes, self.signature_num_bits
        )
        self.embedded_model_hash = self._build_message(signed_model)["model_hash"]
        return signed_model

    def verify_model(self, model: nn.Module) -> Dict[str, Any]:
        """验证模型中嵌入签名与当前模型状态是否匹配。"""
        if (
            self.expected_signature is None
            or self.embedded_signature is None
            or self.expected_bits is None
            or self.signed_message is None
            or self.signature_num_bits is None
        ):
            raise ValueError("No embedded signature available for verification")
        current_message = self._build_message(model)
        extracted_bits = self._extract_signature_bits(model, self.signature_num_bits)
        signature = self.crypto_manager.decode_bits_to_signature(extracted_bits)
        signature_authentic = self.crypto_manager.verify(
            self.signed_message, self.expected_signature
        )
        bits_match = bool(torch.equal(extracted_bits, self.expected_bits))
        hash_matches = current_message["model_hash"] == self.embedded_model_hash
        is_valid = signature_authentic and bits_match and hash_matches
        result: Dict[str, Any] = {
            "is_valid": is_valid,
            "signature": signature,
            "message": current_message,
            "signature_authentic": signature_authentic,
            "bits_match": bits_match,
            "hash_match": hash_matches,
        }
        if "client_id" in self.signed_message:
            result["client_id"] = self.signed_message["client_id"]
        return result

    def export_state(self) -> Dict[str, Any]:
        """导出可序列化的验证状态。"""
        return {
            "public_key": self.crypto_manager.public_key,
            "signature_num_bits": self.signature_num_bits,
            "expected_signature": self.expected_signature,
            "embedded_signature": self.embedded_signature,
            "expected_bits": None
            if self.expected_bits is None
            else self.expected_bits.detach().cpu(),
            "signed_message": self.signed_message,
            "embedded_model_hash": self.embedded_model_hash,
            "strength": self.strength,
            "hash_algorithm": self.hash_algorithm,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """恢复导出的验证状态。"""
        public_key = state.get("public_key")
        if public_key is not None:
            self.crypto_manager.public_key = public_key
        self.signature_num_bits = state.get("signature_num_bits")
        self.expected_signature = state.get("expected_signature")
        self.embedded_signature = state.get("embedded_signature")
        expected_bits = state.get("expected_bits")
        if expected_bits is not None and not isinstance(expected_bits, torch.Tensor):
            expected_bits = torch.tensor(expected_bits, dtype=torch.float32)
        self.expected_bits = expected_bits
        self.signed_message = state.get("signed_message")
        self.embedded_model_hash = state.get("embedded_model_hash")
        saved_strength = state.get("strength")
        if isinstance(saved_strength, (int, float)) and saved_strength > 0:
            self.strength = float(saved_strength)
        saved_hash_algorithm = state.get("hash_algorithm")
        if isinstance(saved_hash_algorithm, str):
            self.hash_algorithm = saved_hash_algorithm
