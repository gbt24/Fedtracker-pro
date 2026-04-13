"""日志管理系统"""

import os
import sys
from datetime import datetime
from typing import Optional


class Logger:
    """日志管理器"""

    def __init__(
        self,
        name: str = "fedtracker",
        log_dir: str = "./logs",
        log_file: Optional[str] = None,
        level: int = 20,  # logging.INFO
        console: bool = True,
    ):
        self.name = name
        self.log_dir = log_dir
        self.level = level

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 生成日志文件名
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"

        self.log_file = os.path.join(log_dir, log_file)

        # 创建logger
        import logging

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # 清除已有handler

        # 文件handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 控制台handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)


# 全局日志实例字典
_global_loggers: dict = {}


def get_logger(name: str = "fedtracker", log_dir: str = "./logs", **kwargs) -> Logger:
    """获取全局日志实例"""
    global _global_loggers
    if name not in _global_loggers:
        _global_loggers[name] = Logger(name, log_dir, **kwargs)
    return _global_loggers[name]
