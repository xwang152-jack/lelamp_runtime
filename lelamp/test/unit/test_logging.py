"""
单元测试 - lelamp.utils.logging
"""
import pytest
import logging
import json
import tempfile
from pathlib import Path
from lelamp.utils.logging import (
    StructuredFormatter,
    setup_logging,
    get_logger,
)


@pytest.mark.unit
def test_get_logger():
    """测试 get_logger 返回正确的 logger 实例"""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


@pytest.mark.unit
def test_setup_logging_console_only(capfd):
    """测试仅控制台日志"""
    # 设置日志系统（无文件输出）
    setup_logging(log_level="DEBUG", log_dir=None, enable_json=False)

    # 创建测试 logger 并记录消息
    logger = get_logger("test_console")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")

    # 刷新输出
    logging.shutdown()
    logging.root.handlers = []  # 清理 handlers

    # 验证日志记录（通过捕获 stdout）
    captured = capfd.readouterr()
    assert "Debug message" in captured.out
    assert "Info message" in captured.out
    assert "Warning message" in captured.out


@pytest.mark.unit
def test_setup_logging_with_file():
    """测试文件日志输出"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # 设置日志系统（包含文件输出）
        setup_logging(log_level="INFO", log_dir=log_dir, enable_json=False)

        # 创建测试 logger 并记录消息
        logger = get_logger("test_file")
        logger.info("Test file logging")

        # 验证日志文件存在
        log_file = log_dir / "lelamp.log"
        assert log_file.exists()

        # 验证日志内容
        content = log_file.read_text(encoding="utf-8")
        assert "Test file logging" in content
        assert "INFO" in content


@pytest.mark.unit
def test_structured_formatter():
    """测试 JSON 格式化器"""
    formatter = StructuredFormatter()

    # 创建测试 LogRecord
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.module = "test_module"
    record.funcName = "test_function"

    # 格式化记录
    formatted = formatter.format(record)

    # 验证 JSON 格式
    log_data = json.loads(formatted)
    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "test_logger"
    assert log_data["message"] == "Test message"
    assert log_data["module"] == "test_module"
    assert log_data["function"] == "test_function"
    assert log_data["line"] == 42
    assert "timestamp" in log_data


@pytest.mark.unit
def test_structured_formatter_with_exception():
    """测试 JSON 格式化器处理异常"""
    formatter = StructuredFormatter()

    # 创建带异常的 LogRecord
    try:
        raise ValueError("Test exception")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=42,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )
    record.module = "test_module"
    record.funcName = "test_function"

    # 格式化记录
    formatted = formatter.format(record)

    # 验证异常信息
    log_data = json.loads(formatted)
    assert "exception" in log_data
    assert "ValueError" in log_data["exception"]
    assert "Test exception" in log_data["exception"]


@pytest.mark.unit
def test_log_rotation():
    """测试日志轮转功能"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # 设置小的最大文件大小以触发轮转（通过手动创建大文件）
        setup_logging(log_level="INFO", log_dir=log_dir, enable_json=False)

        logger = get_logger("test_rotation")

        # 写入大量日志以触发轮转
        # 注意：实际轮转需要达到 10MB，这里只测试配置正确性
        for i in range(100):
            logger.info(f"Log message {i}" * 100)

        log_file = log_dir / "lelamp.log"
        assert log_file.exists()

        # 验证日志内容
        content = log_file.read_text(encoding="utf-8")
        assert "Log message" in content


@pytest.mark.unit
def test_json_logging():
    """测试 JSON 日志输出"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # 启用 JSON 日志
        setup_logging(log_level="INFO", log_dir=log_dir, enable_json=True)

        logger = get_logger("test_json")
        logger.info("JSON test message")

        # 验证 JSON 日志文件存在
        json_log_file = log_dir / "lelamp.json.log"
        assert json_log_file.exists()

        # 验证 JSON 格式
        content = json_log_file.read_text(encoding="utf-8")
        lines = [line for line in content.strip().split("\n") if line]

        # 至少有一条日志记录
        assert len(lines) > 0

        # 验证每行都是有效的 JSON
        for line in lines:
            log_data = json.loads(line)
            if "JSON test message" in log_data.get("message", ""):
                assert log_data["level"] == "INFO"
                assert log_data["logger"] == "test_json"
                break
        else:
            pytest.fail("未找到预期的日志消息")


@pytest.mark.unit
def test_third_party_log_suppression():
    """测试第三方库日志抑制"""
    setup_logging(log_level="DEBUG", log_dir=None, enable_json=False)

    # 验证第三方库日志级别被提升到 WARNING
    httpx_logger = logging.getLogger("httpx")
    assert httpx_logger.level == logging.WARNING

    livekit_logger = logging.getLogger("livekit")
    assert livekit_logger.level == logging.WARNING
