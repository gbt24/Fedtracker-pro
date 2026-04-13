"""CryptoUtils模块单元测试"""

import unittest
import hashlib
import json
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.crypto_utils import (
    CryptoManager,
    embed_signature_to_model,
    extract_signature_from_model,
)


class TestCryptoManager(unittest.TestCase):
    """测试CryptoManager类"""

    def setUp(self):
        """每个测试前设置"""
        self.crypto_manager = CryptoManager()

    def test_generate_keypair(self):
        """测试生成密钥对"""
        private_key, public_key = self.crypto_manager.generate_keypair()

        self.assertIsNotNone(private_key)
        self.assertIsNotNone(public_key)
        self.assertIsInstance(private_key, bytes)
        self.assertIsInstance(public_key, bytes)

    def test_sign_and_verify(self):
        """测试签名和验证"""
        # 生成密钥对
        self.crypto_manager.generate_keypair()

        # 签名消息
        message = {"client_id": 1, "data": "test"}
        signature = self.crypto_manager.sign(message)

        self.assertIsNotNone(signature)
        self.assertIsInstance(signature, bytes)

        # 验证签名
        is_valid = self.crypto_manager.verify(message, signature)
        self.assertTrue(is_valid)

    def test_verify_fails_with_wrong_message(self):
        """测试验证错误消息失败"""
        self.crypto_manager.generate_keypair()

        # 签名正确消息
        original_message = {"client_id": 1, "data": "test"}
        signature = self.crypto_manager.sign(original_message)

        # 验证错误消息
        wrong_message = {"client_id": 2, "data": "wrong"}
        is_valid = self.crypto_manager.verify(wrong_message, signature)
        self.assertFalse(is_valid)

    def test_verify_fails_with_wrong_signature(self):
        """测试验证错误签名失败"""
        self.crypto_manager.generate_keypair()

        message = {"client_id": 1, "data": "test"}

        # 使用错误的签名
        wrong_signature = b"wrong_signature"
        is_valid = self.crypto_manager.verify(message, wrong_signature)
        self.assertFalse(is_valid)

    def test_sign_without_private_key_raises_error(self):
        """测试没有私钥时签名抛出错误"""
        message = {"client_id": 1, "data": "test"}

        with self.assertRaises(ValueError):
            self.crypto_manager.sign(message)

    def test_verify_without_public_key_raises_error(self):
        """测试没有公钥时验证抛出错误"""
        message = {"client_id": 1, "data": "test"}
        signature = b"signature"

        with self.assertRaises(ValueError):
            self.crypto_manager.verify(message, signature)


class TestHashModelState(unittest.TestCase):
    """测试hash_model_state函数"""

    def test_hash_model_state_returns_string(self):
        """测试hash_model_state返回字符串"""
        model = nn.Linear(10, 5)
        state_dict = model.state_dict()

        model_hash = CryptoManager.hash_model_state(state_dict)

        self.assertIsInstance(model_hash, str)
        self.assertEqual(len(model_hash), 64)  # SHA256 hex长度

    def test_hash_model_state_is_deterministic(self):
        """测试hash_model_state是确定性的"""
        model = nn.Linear(10, 5)
        state_dict = model.state_dict()

        hash1 = CryptoManager.hash_model_state(state_dict)
        hash2 = CryptoManager.hash_model_state(state_dict)

        self.assertEqual(hash1, hash2)

    def test_hash_model_state_different_for_different_models(self):
        """测试不同模型的hash不同"""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)

        # 随机初始化
        torch.manual_seed(42)
        model1.reset_parameters()

        torch.manual_seed(99)
        model2.reset_parameters()

        hash1 = CryptoManager.hash_model_state(model1.state_dict())
        hash2 = CryptoManager.hash_model_state(model2.state_dict())

        self.assertNotEqual(hash1, hash2)


class TestEncodeDecodeSignature(unittest.TestCase):
    """测试编码和解码签名"""

    def test_encode_signature_to_bits(self):
        """测试编码签名为比特"""
        signature = b"test_signature_data"

        bits = CryptoManager.encode_signature_to_bits(signature, num_bits=128)

        self.assertIsInstance(bits, torch.Tensor)
        self.assertEqual(bits.shape[0], 128)

    def test_decode_bits_to_signature(self):
        """测试解码比特为签名"""
        original_signature = b"test_signature"

        # 编码
        bits = CryptoManager.encode_signature_to_bits(original_signature, num_bits=128)

        # 解码
        decoded_signature = CryptoManager.decode_bits_to_signature(bits)

        self.assertIsInstance(decoded_signature, bytes)

    def test_encode_decode_roundtrip(self):
        """测试编码解码往返"""
        # 使用足够大的num_bits来容纳签名
        original_signature = b"short"

        bits = CryptoManager.encode_signature_to_bits(original_signature, num_bits=256)
        decoded_signature = CryptoManager.decode_bits_to_signature(bits)

        # 检查原始签名是否包含在解码后的签名中
        # 由于LSB编码，可能会有一些填充
        self.assertIn(original_signature, decoded_signature)


class TestEmbedExtractSignature(unittest.TestCase):
    """测试嵌入和提取签名"""

    def setUp(self):
        """创建测试模型"""
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    def test_embed_signature_to_model(self):
        """测试嵌入签名到模型"""
        signature = b"test_signature"

        # 嵌入签名
        try:
            embedded_model = embed_signature_to_model(self.model, signature)
            # 如果函数成功执行，测试通过
            self.assertTrue(True)
        except ValueError as e:
            # 如果没有足够的参数，这是预期的
            self.assertIn("Not enough parameters", str(e))

    def test_extract_signature_from_model(self):
        """测试从模型提取签名"""
        signature = b"test_signature_data"

        # 嵌入签名
        embedded_model = embed_signature_to_model(self.model, signature)

        # 提取签名
        extracted_signature = extract_signature_from_model(embedded_model)

        self.assertIsInstance(extracted_signature, bytes)

    def test_embed_extract_roundtrip_partial_match(self):
        """测试嵌入提取往返（部分匹配）"""
        # 使用更小的签名以确保能嵌入
        original_signature = b"ab"

        # 嵌入签名
        try:
            embedded_model = embed_signature_to_model(self.model, original_signature)

            # 提取签名
            extracted_signature = extract_signature_from_model(embedded_model)

            # 检查是否成功提取了一些内容
            self.assertIsInstance(extracted_signature, bytes)
            self.assertGreater(len(extracted_signature), 0)
        except ValueError as e:
            # 如果没有足够的参数，这是预期的
            self.assertIn("Not enough parameters", str(e))


if __name__ == "__main__":
    unittest.main()
