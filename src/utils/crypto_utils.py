"""密码学工具"""

import hashlib
import json
from typing import Tuple, Dict, Any, Optional
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256
import torch


class CryptoManager:
    """密码学管理器"""

    def __init__(self, key_size: int = 256, scheme: str = "ecdsa"):
        self.key_size = key_size
        self.scheme = scheme
        self.private_key = None
        self.public_key = None

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """生成密钥对"""
        if self.scheme == "ecdsa":
            # 使用ECC生成密钥对
            key = ECC.generate(curve=f"P-{self.key_size}")
            self.private_key = key.export_key(format="PEM").encode("utf-8")
            self.public_key = key.public_key().export_key(format="PEM").encode("utf-8")
            return self.private_key, self.public_key
        else:
            raise ValueError(f"Unsupported scheme: {self.scheme}")

    def load_keypair(self, private_key_pem: bytes, public_key_pem: bytes):
        """加载密钥对"""
        # 如果是字符串，转换为字节
        if isinstance(private_key_pem, str):
            self.private_key = private_key_pem.encode("utf-8")
        else:
            self.private_key = private_key_pem

        if isinstance(public_key_pem, str):
            self.public_key = public_key_pem.encode("utf-8")
        else:
            self.public_key = public_key_pem

    def sign(self, message: Dict[str, Any]) -> bytes:
        """签名消息"""
        if self.private_key is None:
            raise ValueError("Private key not set")

        # 序列化消息
        message_bytes = json.dumps(message, sort_keys=True).encode("utf-8")

        if self.scheme == "ecdsa":
            # 如果密钥是字节，需要解码为字符串
            if isinstance(self.private_key, bytes):
                self.private_key = self.private_key.decode("utf-8")

            key = ECC.import_key(self.private_key)
            h = SHA256.new(message_bytes)
            signer = DSS.new(key, "fips-186-3")
            signature = signer.sign(h)
            return signature
        else:
            raise ValueError(f"Unsupported scheme: {self.scheme}")

    def verify(self, message: Dict[str, Any], signature: bytes) -> bool:
        """验证签名"""
        if self.public_key is None:
            raise ValueError("Public key not set")

        # 序列化消息
        message_bytes = json.dumps(message, sort_keys=True).encode("utf-8")

        try:
            if self.scheme == "ecdsa":
                # 如果密钥是字节，需要解码为字符串
                if isinstance(self.public_key, bytes):
                    self.public_key = self.public_key.decode("utf-8")

                key = ECC.import_key(self.public_key)
                h = SHA256.new(message_bytes)
                verifier = DSS.new(key, "fips-186-3")
                verifier.verify(h, signature)
                return True
            else:
                raise ValueError(f"Unsupported scheme: {self.scheme}")
        except ValueError:
            return False

    @staticmethod
    def hash_model_state(state_dict: Dict[str, torch.Tensor]) -> str:
        """哈希模型状态"""
        # 将模型状态转换为字节
        state_bytes = b""
        for key in sorted(state_dict.keys()):
            tensor_bytes = state_dict[key].cpu().numpy().tobytes()
            state_bytes += key.encode() + tensor_bytes

        # 计算哈希
        hash_obj = hashlib.sha256(state_bytes)
        return hash_obj.hexdigest()

    @staticmethod
    def encode_signature_to_bits(signature: bytes, num_bits: int = 256) -> torch.Tensor:
        """将签名编码为比特张量"""
        # 将签名转换为整数
        sig_int = int.from_bytes(signature, byteorder="big")

        # 转换为比特
        bits = []
        for i in range(num_bits):
            bits.append((sig_int >> i) & 1)

        return torch.tensor(bits, dtype=torch.float32)

    @staticmethod
    def decode_bits_to_signature(bits: torch.Tensor) -> bytes:
        """将比特张量解码为签名"""
        # 将比特转换为整数
        sig_int = 0
        for i, bit in enumerate(bits):
            sig_int |= int(bit.item()) << i

        # 转换为字节 (假设签名长度不超过512位/64字节)
        num_bytes = (len(bits) + 7) // 8
        return sig_int.to_bytes(num_bytes, byteorder="big")


def embed_signature_to_model(
    model: torch.nn.Module, signature: bytes, strength: float = 0.01
) -> torch.nn.Module:
    """
    将签名嵌入到模型参数中

    使用LSB (Least Significant Bit) 嵌入
    """
    # 将签名转换为比特
    crypto_manager = CryptoManager()
    sig_bits = crypto_manager.encode_signature_to_bits(signature)

    # 获取可嵌入的参数 (权重参数)
    embeddable_params = []
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            embeddable_params.append(param)

    # 确保有足够的参数
    total_elements = sum(p.numel() for p in embeddable_params)
    if total_elements < len(sig_bits):
        raise ValueError(
            f"Not enough parameters to embed signature. "
            f"Need {len(sig_bits)}, have {total_elements}"
        )

    # 嵌入签名比特
    bit_idx = 0
    with torch.no_grad():
        for param in embeddable_params:
            if bit_idx >= len(sig_bits):
                break

            # 展平参数
            flat_param = param.view(-1)
            num_bits = min(len(flat_param), len(sig_bits) - bit_idx)

            # LSB嵌入
            for i in range(num_bits):
                # 获取当前值
                val = flat_param[i].item()

                # 修改LSB
                bit = sig_bits[bit_idx + i].item()
                if bit > 0.5:
                    # 设置LSB为1
                    new_val = (int(val / strength) | 1) * strength
                else:
                    # 设置LSB为0
                    new_val = (int(val / strength) & ~1) * strength

                flat_param[i] = new_val

            bit_idx += num_bits

    return model


def extract_signature_from_model(
    model: torch.nn.Module, num_bits: int = 256, strength: float = 0.01
) -> bytes:
    """从模型参数中提取签名"""
    # 获取可嵌入的参数
    embeddable_params = []
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            embeddable_params.append(param)

    # 提取比特
    bits = []
    for param in embeddable_params:
        if len(bits) >= num_bits:
            break

        flat_param = param.view(-1)
        remaining = num_bits - len(bits)

        for i in range(min(len(flat_param), remaining)):
            val = flat_param[i].item()
            # 提取LSB
            bit = int(val / strength) & 1
            bits.append(float(bit))

    # 转换为签名
    bits_tensor = torch.tensor(bits)
    crypto_manager = CryptoManager()
    signature = crypto_manager.decode_bits_to_signature(bits_tensor)

    return signature
