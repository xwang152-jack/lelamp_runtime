"""
测试设置状态管理器
"""
import os
import tempfile
import pytest
from pathlib import Path


def test_save_and_load_state():
    """测试状态保存和加载"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)
        test_state = {
            "setup_completed": False,
            "current_step": "wifi_selection"
        }

        manager.save_state(test_state)
        loaded_state = manager.load_state()

        assert loaded_state["setup_completed"] == False
        assert loaded_state["current_step"] == "wifi_selection"
    finally:
        os.unlink(state_file)


def test_update_step():
    """测试步骤更新"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)
        manager.update_step("password_input")
        state = manager.load_state()

        assert state["current_step"] == "password_input"
    finally:
        os.unlink(state_file)


def test_complete_setup():
    """测试完成设置"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)
        manager.complete_setup("192.168.1.100")
        state = manager.load_state()

        assert state["setup_completed"] is True
        assert state["last_ip_address"] == "192.168.1.100"
        assert "setup_completed_at" in state
    finally:
        os.unlink(state_file)


def test_increment_attempts():
    """测试增加连接尝试次数"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)

        # 第一次增加
        attempts = manager.increment_attempts()
        assert attempts == 1

        # 第二次增加
        attempts = manager.increment_attempts()
        assert attempts == 2

        # 验证状态
        state = manager.load_state()
        assert state["connection_attempts"] == 2
    finally:
        os.unlink(state_file)


def test_error_handling():
    """测试错误信息处理"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)
        manager.set_error("密码错误")

        state = manager.load_state()
        assert state["error_message"] == "密码错误"

        # 清除错误
        manager.clear_error()
        state = manager.load_state()
        assert state["error_message"] is None
    finally:
        os.unlink(state_file)


def test_is_setup_completed():
    """测试检查设置是否完成"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name

    try:
        from lelamp.api.services.setup_state import SetupStateManager

        manager = SetupStateManager(state_file)

        # 初始状态应该是未完成
        assert manager.is_setup_completed() is False

        # 完成设置
        manager.complete_setup("192.168.1.100")
        assert manager.is_setup_completed() is True
    finally:
        os.unlink(state_file)