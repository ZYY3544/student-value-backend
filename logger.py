"""
===========================================
日志配置模块
===========================================

统一配置日志系统，替代print语句
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回一个logger实例

    Args:
        name: logger名称（通常使用 __name__）
        level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        log_file: 可选的日志文件路径
        format_string: 自定义日志格式

    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # 默认格式：时间 - 模块名 - 级别 - 消息
    if not format_string:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已配置的logger或创建新的

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    logger = logging.getLogger(name)

    # 如果还没有配置，使用默认配置
    if not logger.handlers:
        return setup_logger(name)

    return logger


# 配置根logger（用于整个项目）
def configure_root_logger(
    level: str = "INFO",
    log_dir: Optional[str] = None
):
    """
    配置项目的根logger

    Args:
        level: 日志级别
        log_dir: 日志目录（如果提供，会创建日志文件）
    """
    log_file = None
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = str(log_path / 'hay_evaluation.log')

    setup_logger(
        'hay_evaluation',
        level=level,
        log_file=log_file
    )


# 为各模块提供便捷函数
def get_module_logger(module_name: str) -> logging.Logger:
    """
    为指定模块获取logger

    Usage:
        from logger import get_module_logger
        logger = get_module_logger(__name__)
        logger.info("模块初始化完成")
    """
    return get_logger(f'hay_evaluation.{module_name}')
