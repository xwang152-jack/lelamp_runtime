"""
增强的日志系统 - 支持结构化日志和轮转
"""
import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON 格式化器，用于结构化日志"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_json: bool = False,
) -> None:
    """
    设置日志系统

    Args:
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_dir: 日志目录路径，如果为 None 则不输出到文件
        enable_json: 是否启用 JSON 格式日志
    """
    # 获取 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 清除已有的 handlers
    root_logger.handlers.clear()

    # 控制台 handler（人类可读格式）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(root_logger.level)
    console_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件 handler（如果指定了日志目录）
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 标准文本日志
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "lelamp.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(root_logger.level)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)

        # JSON 日志（如果启用）
        if enable_json:
            json_handler = logging.handlers.RotatingFileHandler(
                log_dir / "lelamp.json.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            json_handler.setLevel(root_logger.level)
            json_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(json_handler)

    # 抑制第三方库的日志
    _suppress_third_party_logs()


def _suppress_third_party_logs() -> None:
    """抑制第三方库的日志输出"""
    # 常见的啰嗦第三方库
    noisy_loggers = [
        "httpx",
        "httpcore",
        "livekit",
        "urllib3",
        "asyncio",
        "multipart",
        "pyav",
        "libav",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger

    Args:
        name: logger 名称

    Returns:
        logging.Logger 实例
    """
    return logging.getLogger(name)
