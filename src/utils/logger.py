"""
日志工具
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "nlp-model-lab",
    log_file: str = None,
    level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
        console: 是否输出到控制台

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 清除现有处理器
    logger.handlers.clear()

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_log_dir(base_dir: str = "./outputs/logs") -> str:
    """
    获取日志目录路径（带时间戳）

    Args:
        base_dir: 基础日志目录

    Returns:
        日志目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)