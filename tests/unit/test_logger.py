"""Logger模块单元测试"""

import unittest
import os
import tempfile
import shutil
import logging
from pathlib import Path
import sys

# 添加src到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.logger import Logger, get_logger


class TestLogger(unittest.TestCase):
    """测试Logger类"""

    def setUp(self):
        """每个测试前设置临时目录"""
        self.test_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.test_dir, "logs")

    def tearDown(self):
        """每个测试后清理临时目录"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        # 清除全局logger
        import src.utils.logger

        src.utils.logger._global_logger = None

    def test_logger_creates_log_directory(self):
        """测试Logger创建日志目录"""
        logger = Logger(name="test", log_dir=self.log_dir)
        self.assertTrue(os.path.exists(self.log_dir))

    def test_logger_creates_log_file(self):
        """测试Logger创建日志文件"""
        logger = Logger(name="test", log_dir=self.log_dir)
        self.assertTrue(os.path.exists(logger.log_file))

    def test_logger_info_message(self):
        """测试Logger记录info消息"""
        logger = Logger(name="test", log_dir=self.log_dir, console=False)
        logger.info("Test info message")

        # 验证日志文件包含消息
        with open(logger.log_file, "r") as f:
            content = f.read()
            self.assertIn("Test info message", content)
            self.assertIn("INFO", content)

    def test_logger_warning_message(self):
        """测试Logger记录warning消息"""
        logger = Logger(name="test", log_dir=self.log_dir, console=False)
        logger.warning("Test warning message")

        with open(logger.log_file, "r") as f:
            content = f.read()
            self.assertIn("Test warning message", content)
            self.assertIn("WARNING", content)

    def test_logger_error_message(self):
        """测试Logger记录error消息"""
        logger = Logger(name="test", log_dir=self.log_dir, console=False)
        logger.error("Test error message")

        with open(logger.log_file, "r") as f:
            content = f.read()
            self.assertIn("Test error message", content)
            self.assertIn("ERROR", content)

    def test_logger_debug_message(self):
        """测试Logger记录debug消息"""
        logger = Logger(
            name="test", log_dir=self.log_dir, console=False, level=logging.DEBUG
        )
        logger.debug("Test debug message")

        with open(logger.log_file, "r") as f:
            content = f.read()
            self.assertIn("Test debug message", content)
            self.assertIn("DEBUG", content)

    def test_logger_with_custom_log_file(self):
        """测试Logger使用自定义日志文件名"""
        custom_file = "custom_test.log"
        logger = Logger(name="test", log_dir=self.log_dir, log_file=custom_file)
        expected_path = os.path.join(self.log_dir, custom_file)
        self.assertEqual(logger.log_file, expected_path)

    def test_get_logger_returns_singleton(self):
        """测试get_logger返回单例"""
        logger1 = get_logger(name="test", log_dir=self.log_dir, console=False)
        logger2 = get_logger(name="test", log_dir=self.log_dir, console=False)
        self.assertIs(logger1, logger2)

    def test_get_logger_with_different_names_creates_different_instances(self):
        """测试不同名称的logger创建不同实例"""
        logger1 = get_logger(name="test1", log_dir=self.log_dir, console=False)
        logger2 = get_logger(name="test2", log_dir=self.log_dir, console=False)
        self.assertIsNot(logger1, logger2)

    def test_logger_console_output_disabled(self):
        """测试禁用控制台输出"""
        # 这个测试主要验证不会抛出异常
        logger = Logger(name="test", log_dir=self.log_dir, console=False)
        logger.info("This should not print to console")
        # 如果没有抛出异常，则测试通过
        self.assertTrue(True)

    def test_logger_log_file_timestamp(self):
        """测试日志文件名包含时间戳"""
        logger = Logger(name="test", log_dir=self.log_dir)
        # 日志文件应该包含时间戳
        log_filename = os.path.basename(logger.log_file)
        self.assertIn("test_", log_filename)


if __name__ == "__main__":
    unittest.main()
