# LeLamp Runtime 项目优化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重构 LeLamp Runtime 项目，建立测试体系、模块化架构、现代化前端和 CI/CD 流程

**Architecture:** 分四个阶段渐进式重构：(1) 测试基础设施，(2) 代码重构（main.py 瘦身），(3) 前端现代化（Vue 3），(4) CI/CD 集成

**Tech Stack:** Python 3.12, pytest, Vue 3, TypeScript, Vite, Pinia, GitHub Actions, Ruff

---

## Phase 1: 测试基础设施（第一阶段）

### Task 1.1: 配置 pytest 和开发依赖

**Files:**
- Modify: `pyproject.toml`
- Create: `pytest.ini`
- Create: `tests/conftest.py`

**Step 1: 添加开发依赖到 pyproject.toml**

在 `pyproject.toml` 的 `[project.optional-dependencies]` 部分添加：

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0,<9.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "pytest-asyncio>=0.23.0,<1.0.0",
    "pytest-mock>=3.12.0,<4.0.0",
    "ruff>=0.3.0,<1.0.0",
]
```

**Step 2: 创建 pytest.ini 配置文件**

```ini
[pytest]
testpaths = tests lelamp/test
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=lelamp
    --cov-report=term-missing
    --cov-report=html
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (may use mocks)
    hardware: Hardware-dependent tests (requires actual hardware)
    slow: Slow tests
```

**Step 3: 创建 tests/conftest.py 共享 fixtures**

```python
"""
Pytest 共享配置和 fixtures
"""
import pytest
from unittest.mock import Mock
from lelamp.config import AppConfig


@pytest.fixture
def mock_config():
    """提供测试用配置"""
    return AppConfig(
        livekit_url="wss://test.livekit.io",
        livekit_api_key="test_key",
        livekit_api_secret="test_secret",
        deepseek_model="deepseek-chat",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_api_key="test_deepseek_key",
        modelscope_base_url="https://api-inference.modelscope.cn/v1",
        modelscope_api_key=None,
        modelscope_model="Qwen/Qwen3-VL-235B-A22B-Instruct",
        modelscope_timeout_s=60.0,
        vision_enabled=True,
        camera_index_or_path=0,
        camera_width=1024,
        camera_height=768,
        vision_capture_interval_s=2.5,
        vision_jpeg_quality=92,
        vision_max_age_s=15.0,
    )


@pytest.fixture
def mock_motors_service():
    """Mock MotorsService"""
    mock = Mock()
    mock.is_running = True
    mock.has_pending_event.return_value = False
    return mock


@pytest.fixture
def mock_rgb_service():
    """Mock RGBService"""
    mock = Mock()
    mock.is_running = True
    return mock


@pytest.fixture
def mock_vision_service():
    """Mock VisionService"""
    mock = Mock()
    mock.is_running = True
    return mock
```

**Step 4: 安装开发依赖**

Run: `uv sync --extra dev`
Expected: 安装成功，输出类似 "Resolved X packages in Ys"

**Step 5: 验证 pytest 配置**

Run: `uv run pytest --collect-only`
Expected: 收集到现有测试（test_security.py, test_url_validation.py, test_ota.py）

**Step 6: Commit**

```bash
git add pyproject.toml pytest.ini tests/conftest.py
git commit -m "test: configure pytest and dev dependencies

- Add pytest, pytest-cov, pytest-asyncio, pytest-mock, ruff
- Create pytest.ini with test markers (unit/integration/hardware)
- Add shared fixtures in tests/conftest.py"
```

---

### Task 1.2: 重组测试目录结构

**Files:**
- Create: `lelamp/test/unit/__init__.py`
- Create: `lelamp/test/integration/__init__.py`
- Create: `lelamp/test/hardware/__init__.py`
- Move: `lelamp/test/test_security.py` → `lelamp/test/unit/test_security.py`
- Move: `lelamp/test/test_url_validation.py` → `lelamp/test/unit/test_url_validation.py`
- Move: `lelamp/test/test_ota.py` → `lelamp/test/unit/test_ota.py`
- Move: `lelamp/test/test_motors.py` → `lelamp/test/hardware/test_motors.py`
- Move: `lelamp/test/test_rgb.py` → `lelamp/test/hardware/test_rgb.py`
- Move: `lelamp/test/test_audio.py` → `lelamp/test/hardware/test_audio.py`

**Step 1: 创建测试子目录和 __init__.py**

```bash
mkdir -p lelamp/test/unit lelamp/test/integration lelamp/test/hardware
touch lelamp/test/unit/__init__.py
touch lelamp/test/integration/__init__.py
touch lelamp/test/hardware/__init__.py
```

**Step 2: 移动单元测试**

```bash
git mv lelamp/test/test_security.py lelamp/test/unit/
git mv lelamp/test/test_url_validation.py lelamp/test/unit/
git mv lelamp/test/test_ota.py lelamp/test/unit/
```

**Step 3: 移动硬件测试**

```bash
git mv lelamp/test/test_motors.py lelamp/test/hardware/
git mv lelamp/test/test_rgb.py lelamp/test/hardware/
git mv lelamp/test/test_audio.py lelamp/test/hardware/
```

**Step 4: 更新硬件测试添加 @pytest.mark.hardware**

在每个硬件测试文件顶部的测试函数添加装饰器（示例 - `lelamp/test/hardware/test_motors.py`）：

```python
import pytest

@pytest.mark.hardware
def test_find_port():
    # 现有测试代码
    pass
```

**Step 5: 验证测试收集**

Run: `uv run pytest --collect-only`
Expected: 收集到 unit/ 和 hardware/ 下的测试

**Step 6: 运行单元测试验证**

Run: `uv run pytest -m unit -v`
Expected: 单元测试全部通过

**Step 7: Commit**

```bash
git add lelamp/test/
git commit -m "test: reorganize tests into unit/integration/hardware structure

- Create unit/, integration/, hardware/ subdirectories
- Move test_security, test_url_validation, test_ota to unit/
- Move test_motors, test_rgb, test_audio to hardware/
- Add @pytest.mark.hardware to hardware tests"
```

---

### Task 1.3: 添加 rate_limiter 单元测试

**Files:**
- Create: `lelamp/test/unit/test_rate_limiter.py`

**Step 1: 编写 rate_limiter 测试**

```python
"""
Rate Limiter 单元测试
"""
import pytest
import time
from lelamp.utils.rate_limiter import RateLimiter, get_rate_limiter


@pytest.mark.unit
class TestRateLimiter:
    """RateLimiter 测试套件"""

    def test_basic_acquisition(self):
        """测试基本令牌获取"""
        limiter = RateLimiter(rate=2.0, capacity=2)  # 2 req/s, 容量 2
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False  # 超出容量

    def test_token_refill(self):
        """测试令牌补充"""
        limiter = RateLimiter(rate=10.0, capacity=1)  # 10 tokens/s
        assert limiter.try_acquire() is True
        time.sleep(0.15)  # 等待 150ms，应该补充 1.5 个令牌
        assert limiter.try_acquire() is True  # 现在应该有令牌

    def test_zero_capacity(self):
        """测试零容量场景"""
        limiter = RateLimiter(rate=1.0, capacity=0)
        assert limiter.try_acquire() is False

    def test_get_rate_limiter_singleton(self):
        """测试 get_rate_limiter 返回同一实例"""
        limiter1 = get_rate_limiter("test_api", rate=1.0, capacity=1)
        limiter2 = get_rate_limiter("test_api", rate=1.0, capacity=1)
        assert limiter1 is limiter2

    def test_different_limiters(self):
        """测试不同名称返回不同实例"""
        limiter1 = get_rate_limiter("api1", rate=1.0, capacity=1)
        limiter2 = get_rate_limiter("api2", rate=1.0, capacity=1)
        assert limiter1 is not limiter2

    def test_statistics(self):
        """测试统计信息"""
        limiter = RateLimiter(rate=10.0, capacity=2)
        limiter.try_acquire()
        limiter.try_acquire()
        limiter.try_acquire()  # 这次会失败

        stats = limiter.get_stats()
        assert stats["requests"] == 3
        assert stats["allowed"] == 2
        assert stats["denied"] == 1
```

**Step 2: 运行测试验证**

Run: `uv run pytest lelamp/test/unit/test_rate_limiter.py -v`
Expected: 所有测试通过

**Step 3: Commit**

```bash
git add lelamp/test/unit/test_rate_limiter.py
git commit -m "test: add unit tests for rate_limiter

- Test basic token acquisition
- Test token refill mechanism
- Test singleton pattern for get_rate_limiter
- Test statistics tracking"
```

---

### Task 1.4: 添加 cache_manager 单元测试

**Files:**
- Create: `lelamp/test/unit/test_cache.py`

**Step 1: 编写 cache_manager 测试**

```python
"""
Cache Manager 单元测试
"""
import pytest
import time
from lelamp.cache.cache_manager import VisionCache, SearchCache


@pytest.mark.unit
class TestVisionCache:
    """VisionCache 测试套件"""

    def test_basic_set_get(self):
        """测试基本的存取操作"""
        cache = VisionCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """测试 TTL 过期"""
        cache = VisionCache(default_ttl=0.2)  # 200ms TTL
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        time.sleep(0.3)  # 等待超过 TTL
        assert cache.get("key1") is None  # 应该已过期

    def test_custom_ttl(self):
        """测试自定义 TTL"""
        cache = VisionCache(default_ttl=10.0)
        cache.set("key1", "value1", ttl=0.2)  # 自定义 200ms TTL

        time.sleep(0.3)
        assert cache.get("key1") is None

    def test_overwrite_existing_key(self):
        """测试覆盖已存在的键"""
        cache = VisionCache()
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_clear(self):
        """测试清空缓存"""
        cache = VisionCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_lru_eviction(self):
        """测试 LRU 淘汰策略"""
        cache = VisionCache(max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # 应该淘汰 key1

        assert cache.get("key1") is None  # 已被淘汰
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"


@pytest.mark.unit
class TestSearchCache:
    """SearchCache 测试套件"""

    def test_basic_operations(self):
        """测试基本操作"""
        cache = SearchCache()
        cache.set("query1", {"results": ["result1"]})
        result = cache.get("query1")
        assert result == {"results": ["result1"]}

    def test_different_cache_instances(self):
        """测试 VisionCache 和 SearchCache 独立"""
        vision = VisionCache()
        search = SearchCache()

        vision.set("key1", "vision_value")
        search.set("key1", "search_value")

        assert vision.get("key1") == "vision_value"
        assert search.get("key1") == "search_value"
```

**Step 2: 运行测试验证**

Run: `uv run pytest lelamp/test/unit/test_cache.py -v`
Expected: 所有测试通过

**Step 3: Commit**

```bash
git add lelamp/test/unit/test_cache.py
git commit -m "test: add unit tests for cache_manager

- Test basic set/get operations
- Test TTL expiration and custom TTL
- Test cache clear functionality
- Test LRU eviction policy
- Test VisionCache and SearchCache independence"
```

---

### Task 1.5: 添加 config 单元测试

**Files:**
- Create: `lelamp/test/unit/test_config.py`

**Step 1: 编写 config 测试**

```python
"""
Config 单元测试
"""
import pytest
import os
from unittest.mock import patch
from lelamp.config import (
    _get_env_str,
    _get_env_bool,
    _get_env_int,
    _get_env_float,
    _require_env,
    _parse_index_or_path,
    AppConfig,
)


@pytest.mark.unit
class TestConfigHelpers:
    """配置辅助函数测试"""

    def test_get_env_str_with_default(self):
        """测试字符串环境变量获取"""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_str("MISSING_VAR", "default") == "default"

        with patch.dict(os.environ, {"TEST_VAR": "value"}, clear=True):
            assert _get_env_str("TEST_VAR", "default") == "value"

    def test_get_env_bool(self):
        """测试布尔环境变量解析"""
        test_cases = {
            "1": True,
            "true": True,
            "True": True,
            "TRUE": True,
            "yes": True,
            "0": False,
            "false": False,
            "False": False,
            "no": False,
        }

        for value, expected in test_cases.items():
            with patch.dict(os.environ, {"TEST_BOOL": value}, clear=True):
                assert _get_env_bool("TEST_BOOL", False) == expected

    def test_get_env_int(self):
        """测试整数环境变量解析"""
        with patch.dict(os.environ, {"TEST_INT": "42"}, clear=True):
            assert _get_env_int("TEST_INT", 0) == 42

        with patch.dict(os.environ, {"TEST_INT": "invalid"}, clear=True):
            assert _get_env_int("TEST_INT", 10) == 10  # 回退到默认值

    def test_get_env_float(self):
        """测试浮点数环境变量解析"""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}, clear=True):
            assert _get_env_float("TEST_FLOAT", 0.0) == 3.14

        with patch.dict(os.environ, {"TEST_FLOAT": "invalid"}, clear=True):
            assert _get_env_float("TEST_FLOAT", 1.0) == 1.0

    def test_require_env_present(self):
        """测试必需环境变量存在"""
        with patch.dict(os.environ, {"REQUIRED_VAR": "value"}, clear=True):
            assert _require_env("REQUIRED_VAR") == "value"

    def test_require_env_missing(self):
        """测试必需环境变量缺失抛出异常"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Required environment variable"):
                _require_env("MISSING_REQUIRED")

    def test_parse_index_or_path_integer(self):
        """测试解析整数索引"""
        assert _parse_index_or_path("0") == 0
        assert _parse_index_or_path("2") == 2

    def test_parse_index_or_path_string(self):
        """测试解析字符串路径"""
        assert _parse_index_or_path("/dev/video0") == "/dev/video0"
        assert _parse_index_or_path("rtsp://camera") == "rtsp://camera"


@pytest.mark.unit
class TestAppConfig:
    """AppConfig 测试"""

    def test_app_config_immutable(self):
        """测试 AppConfig 是冻结的（不可变）"""
        config = AppConfig(
            livekit_url="wss://test",
            livekit_api_key="key",
            livekit_api_secret="secret",
            deepseek_model="model",
            deepseek_base_url="url",
            deepseek_api_key="key",
            modelscope_base_url="url",
            modelscope_api_key=None,
            modelscope_model="model",
            modelscope_timeout_s=60.0,
            vision_enabled=True,
            camera_index_or_path=0,
            camera_width=1024,
            camera_height=768,
            vision_capture_interval_s=2.5,
            vision_jpeg_quality=92,
            vision_max_age_s=15.0,
        )

        with pytest.raises(AttributeError):
            config.livekit_url = "new_url"  # 应该失败
```

**Step 2: 运行测试验证**

Run: `uv run pytest lelamp/test/unit/test_config.py -v`
Expected: 所有测试通过

**Step 3: Commit**

```bash
git add lelamp/test/unit/test_config.py
git commit -m "test: add unit tests for config module

- Test environment variable helpers (_get_env_*)
- Test _require_env exception handling
- Test _parse_index_or_path for camera config
- Test AppConfig immutability"
```

---

## Phase 2: 代码重构（第二阶段）

### Task 2.1: 创建 lelamp/agent/states.py

**Files:**
- Create: `lelamp/agent/__init__.py` (更新)
- Create: `lelamp/agent/states.py`

**Step 1: 创建 states.py 定义会话状态**

```python
"""
会话状态管理
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time
import threading


class ConversationState(str, Enum):
    """会话状态枚举"""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class StateColors:
    """状态对应的 RGB 颜色"""
    IDLE = (255, 244, 229)      # 暖白光
    LISTENING = (0, 140, 255)   # 蓝色
    THINKING = (180, 0, 255)    # 紫色
    # SPEAKING 使用随机动画颜色


class StateManager:
    """状态管理器 - 管理会话状态、冷却时间和覆盖逻辑"""

    def __init__(
        self,
        motion_cooldown_s: float = 3.0,
        suppress_motion_after_light_s: float = 5.0,
    ):
        self._current_state = ConversationState.IDLE
        self._motion_cooldown_s = motion_cooldown_s
        self._suppress_motion_after_light_s = suppress_motion_after_light_s

        # 时间戳追踪
        self._last_motion_ts: Optional[float] = None
        self._light_override_until_ts: Optional[float] = None
        self._suppress_motion_until_ts: Optional[float] = None

        # 线程锁
        self._state_lock = threading.Lock()
        self._timestamps_lock = threading.Lock()

    @property
    def current_state(self) -> ConversationState:
        """获取当前状态"""
        with self._state_lock:
            return self._current_state

    def set_state(self, state: ConversationState) -> None:
        """设置当前状态"""
        with self._state_lock:
            self._current_state = state

    def can_execute_motion(self) -> bool:
        """检查是否允许执行电机动作"""
        with self._timestamps_lock:
            now = time.time()

            # 检查是否在抑制期内
            if (self._suppress_motion_until_ts is not None and
                now < self._suppress_motion_until_ts):
                return False

            # 检查冷却时间
            if (self._last_motion_ts is not None and
                (now - self._last_motion_ts) < self._motion_cooldown_s):
                return False

            return True

    def record_motion(self) -> None:
        """记录电机动作时间"""
        with self._timestamps_lock:
            self._last_motion_ts = time.time()

    def is_light_overridden(self) -> bool:
        """检查灯光是否被手动覆盖"""
        with self._timestamps_lock:
            if self._light_override_until_ts is None:
                return False
            return time.time() < self._light_override_until_ts

    def set_light_override(self, duration_s: float) -> None:
        """设置灯光覆盖"""
        with self._timestamps_lock:
            self._light_override_until_ts = time.time() + duration_s
            self._suppress_motion_until_ts = time.time() + self._suppress_motion_after_light_s

    def clear_light_override(self) -> None:
        """清除灯光覆盖"""
        with self._timestamps_lock:
            self._light_override_until_ts = None
```

**Step 2: 更新 lelamp/agent/__init__.py**

```python
"""
Agent 模块 - 导出核心类
"""
from .states import ConversationState, StateColors, StateManager

__all__ = [
    "ConversationState",
    "StateColors",
    "StateManager",
]
```

**Step 3: 编写 states 单元测试**

Create: `lelamp/test/unit/test_states.py`

```python
"""
States 模块单元测试
"""
import pytest
import time
from lelamp.agent.states import ConversationState, StateManager


@pytest.mark.unit
class TestStateManager:
    """StateManager 测试套件"""

    def test_initial_state(self):
        """测试初始状态"""
        manager = StateManager()
        assert manager.current_state == ConversationState.IDLE

    def test_set_state(self):
        """测试状态切换"""
        manager = StateManager()
        manager.set_state(ConversationState.LISTENING)
        assert manager.current_state == ConversationState.LISTENING

    def test_motion_cooldown(self):
        """测试电机冷却时间"""
        manager = StateManager(motion_cooldown_s=0.2)

        # 第一次应该允许
        assert manager.can_execute_motion() is True
        manager.record_motion()

        # 立即再次尝试应该被拒绝
        assert manager.can_execute_motion() is False

        # 等待冷却时间后应该允许
        time.sleep(0.25)
        assert manager.can_execute_motion() is True

    def test_light_override(self):
        """测试灯光覆盖"""
        manager = StateManager()

        # 初始不应该被覆盖
        assert manager.is_light_overridden() is False

        # 设置覆盖
        manager.set_light_override(duration_s=0.2)
        assert manager.is_light_overridden() is True

        # 等待过期
        time.sleep(0.25)
        assert manager.is_light_overridden() is False

    def test_suppress_motion_after_light(self):
        """测试灯光命令后抑制电机动作"""
        manager = StateManager(suppress_motion_after_light_s=0.2)

        # 设置灯光覆盖
        manager.set_light_override(duration_s=0.1)

        # 在抑制期内不允许电机动作
        assert manager.can_execute_motion() is False

        # 等待抑制期结束
        time.sleep(0.25)
        assert manager.can_execute_motion() is True

    def test_clear_light_override(self):
        """测试清除灯光覆盖"""
        manager = StateManager()
        manager.set_light_override(duration_s=10.0)
        assert manager.is_light_overridden() is True

        manager.clear_light_override()
        assert manager.is_light_overridden() is False
```

**Step 4: 运行测试验证**

Run: `uv run pytest lelamp/test/unit/test_states.py -v`
Expected: 所有测试通过

**Step 5: Commit**

```bash
git add lelamp/agent/states.py lelamp/agent/__init__.py lelamp/test/unit/test_states.py
git commit -m "refactor: create agent states management module

- Add ConversationState enum and StateColors
- Create StateManager for state, cooldown, and override logic
- Thread-safe state and timestamp management
- Add comprehensive unit tests for StateManager"
```

---

### Task 2.2: 创建 lelamp/agent/tools/motor_tools.py

**Files:**
- Create: `lelamp/agent/tools/__init__.py`
- Create: `lelamp/agent/tools/motor_tools.py`
- Reference: `main.py` (提取 function_tool 代码)

**Step 1: 从 main.py 提取电机相关的 function_tool**

创建 `lelamp/agent/tools/motor_tools.py`:

```python
"""
电机控制工具
"""
import logging
from typing import TYPE_CHECKING
from livekit.agents import function_tool

if TYPE_CHECKING:
    from lelamp.service.motors.motors_service import MotorsService
    from lelamp.agent.states import StateManager

logger = logging.getLogger(__name__)

# 定义关节安全角度范围
SAFE_JOINT_RANGES = {
    "base_yaw": (-180, 180),
    "base_pitch": (-90, 90),
    "elbow_pitch": (-150, 150),
    "wrist_roll": (-180, 180),
    "wrist_pitch": (-90, 90),
}


class MotorTools:
    """电机控制工具类"""

    def __init__(self, motors_service: "MotorsService", state_manager: "StateManager"):
        self.motors = motors_service
        self.state_manager = state_manager

    @function_tool
    def play_recording(self, name: str) -> str:
        """
        播放预录制的电机动作序列

        Args:
            name: 录制文件名（不含 .csv 后缀），如 "wave", "nod", "shake_head"

        Returns:
            执行结果描述
        """
        if not self.state_manager.can_execute_motion():
            logger.info(f"Motion suppressed due to cooldown or light override")
            return f"动作被抑制（冷却中或灯光控制中）"

        from lelamp.service import Priority
        self.motors.dispatch("play_recording", {"name": name}, Priority.HIGH)
        self.state_manager.record_motion()
        logger.info(f"Playing recording: {name}")
        return f"开始播放动作: {name}"

    @function_tool
    def move_joint(self, joint: str, position: float) -> str:
        """
        移动单个关节到指定位置

        Args:
            joint: 关节名称（base_yaw, base_pitch, elbow_pitch, wrist_roll, wrist_pitch）
            position: 目标角度（度）

        Returns:
            执行结果描述
        """
        # 验证关节名称
        if joint not in SAFE_JOINT_RANGES:
            valid_joints = ", ".join(SAFE_JOINT_RANGES.keys())
            return f"错误：未知关节 '{joint}'。有效关节：{valid_joints}"

        # 验证角度范围
        min_angle, max_angle = SAFE_JOINT_RANGES[joint]
        if not (min_angle <= position <= max_angle):
            return f"错误：关节 '{joint}' 角度 {position}° 超出安全范围 [{min_angle}, {max_angle}]"

        if not self.state_manager.can_execute_motion():
            logger.info(f"Motion suppressed for move_joint {joint}={position}")
            return f"动作被抑制（冷却中或灯光控制中）"

        from lelamp.service import Priority
        self.motors.dispatch("move_joint", {"joint": joint, "position": position}, Priority.NORMAL)
        self.state_manager.record_motion()
        logger.info(f"Moving joint {joint} to {position}°")
        return f"移动关节 {joint} 到 {position}°"

    @function_tool
    def get_joint_positions(self) -> str:
        """
        获取所有关节的当前位置

        Returns:
            当前关节位置的字符串描述
        """
        # 注意：这需要 MotorsService 实现 get_positions 方法
        # 这里提供一个占位实现
        return "当前关节位置查询功能待实现"

    @function_tool
    def get_motor_health(self) -> str:
        """
        获取电机健康状态摘要

        Returns:
            电机健康状态描述
        """
        if not hasattr(self.motors, 'get_motor_health_summary'):
            return "电机健康监控未启用"

        health_summary = self.motors.get_motor_health_summary()

        # 格式化输出
        result = ["电机健康状态:"]
        for motor_name, health in health_summary.items():
            state = health.get("state", "UNKNOWN")
            temp = health.get("temperature_c", "N/A")
            voltage = health.get("voltage_v", "N/A")
            result.append(f"  {motor_name}: {state} (温度: {temp}°C, 电压: {voltage}V)")

        return "\n".join(result)
```

**Step 2: 创建 tools/__init__.py**

```python
"""
Agent 工具模块
"""
from .motor_tools import MotorTools

__all__ = ["MotorTools"]
```

**Step 3: 编写 motor_tools 集成测试**

Create: `lelamp/test/integration/test_motor_tools.py`

```python
"""
MotorTools 集成测试
"""
import pytest
from unittest.mock import Mock
from lelamp.agent.tools.motor_tools import MotorTools, SAFE_JOINT_RANGES
from lelamp.agent.states import StateManager


@pytest.mark.integration
class TestMotorTools:
    """MotorTools 集成测试套件"""

    @pytest.fixture
    def motor_tools(self):
        """创建 MotorTools 实例"""
        motors_service = Mock()
        state_manager = StateManager()
        return MotorTools(motors_service, state_manager)

    def test_play_recording_success(self, motor_tools):
        """测试播放录制动作"""
        result = motor_tools.play_recording("wave")
        assert "wave" in result
        motor_tools.motors.dispatch.assert_called_once()

    def test_play_recording_cooldown(self, motor_tools):
        """测试冷却期阻止动作"""
        motor_tools.play_recording("wave")
        result = motor_tools.play_recording("nod")  # 立即再次调用
        assert "抑制" in result or "冷却" in result

    def test_move_joint_valid(self, motor_tools):
        """测试移动关节到有效位置"""
        result = motor_tools.move_joint("base_yaw", 45.0)
        assert "base_yaw" in result
        assert "45" in result
        motor_tools.motors.dispatch.assert_called_once()

    def test_move_joint_invalid_name(self, motor_tools):
        """测试无效关节名称"""
        result = motor_tools.move_joint("invalid_joint", 0.0)
        assert "错误" in result
        assert "未知关节" in result

    def test_move_joint_out_of_range(self, motor_tools):
        """测试角度超出安全范围"""
        result = motor_tools.move_joint("base_yaw", 200.0)  # 超出 [-180, 180]
        assert "错误" in result
        assert "超出安全范围" in result

    def test_safe_joint_ranges_coverage(self):
        """测试所有关节都有安全范围定义"""
        expected_joints = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]
        for joint in expected_joints:
            assert joint in SAFE_JOINT_RANGES
            min_angle, max_angle = SAFE_JOINT_RANGES[joint]
            assert min_angle < max_angle
```

**Step 4: 运行测试验证**

Run: `uv run pytest lelamp/test/integration/test_motor_tools.py -v`
Expected: 所有测试通过

**Step 5: Commit**

```bash
git add lelamp/agent/tools/ lelamp/test/integration/test_motor_tools.py
git commit -m "refactor: extract motor tools from main.py

- Create MotorTools class with play_recording, move_joint, get_motor_health
- Implement SAFE_JOINT_RANGES validation
- Integrate with StateManager for cooldown/override logic
- Add comprehensive integration tests"
```

---

### Task 2.3: 创建 lelamp/agent/tools/rgb_tools.py

**Files:**
- Create: `lelamp/agent/tools/rgb_tools.py`
- Update: `lelamp/agent/tools/__init__.py`

**Step 1: 从 main.py 提取 RGB 相关的 function_tool**

```python
"""
RGB 灯光控制工具
"""
import logging
import random
from typing import TYPE_CHECKING
from livekit.agents import function_tool

if TYPE_CHECKING:
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.agent.states import StateManager

logger = logging.getLogger(__name__)


class RGBTools:
    """RGB 灯光控制工具类"""

    def __init__(self, rgb_service: "RGBService", state_manager: "StateManager"):
        self.rgb = rgb_service
        self.state_manager = state_manager

    @function_tool
    def set_rgb_solid(self, r: int, g: int, b: int) -> str:
        """
        设置纯色灯光

        Args:
            r: 红色值 (0-255)
            g: 绿色值 (0-255)
            b: 蓝色值 (0-255)

        Returns:
            执行结果描述
        """
        # 验证颜色范围
        if not all(0 <= val <= 255 for val in [r, g, b]):
            return f"错误：RGB 值必须在 0-255 范围内"

        from lelamp.service import Priority
        self.rgb.dispatch("solid", {"color": (r, g, b)}, Priority.HIGH)
        self.state_manager.set_light_override(duration_s=10.0)
        logger.info(f"Set RGB solid color: ({r}, {g}, {b})")
        return f"设置纯色灯光: RGB({r}, {g}, {b})"

    @function_tool
    def paint_rgb_pattern(self, pattern: str) -> str:
        """
        绘制预定义的 LED 图案

        Args:
            pattern: 图案名称（如 "heart", "smile", "arrow"）

        Returns:
            执行结果描述
        """
        from lelamp.service import Priority
        self.rgb.dispatch("pattern", {"pattern": pattern}, Priority.NORMAL)
        self.state_manager.set_light_override(duration_s=10.0)
        logger.info(f"Paint RGB pattern: {pattern}")
        return f"绘制图案: {pattern}"

    @function_tool
    def rgb_effect_rainbow(self, speed: float = 1.0) -> str:
        """
        启动彩虹效果

        Args:
            speed: 速度倍率（默认 1.0）

        Returns:
            执行结果描述
        """
        from lelamp.service import Priority
        self.rgb.dispatch("effect", {"name": "rainbow", "speed": speed}, Priority.NORMAL)
        self.state_manager.set_light_override(duration_s=15.0)
        logger.info(f"Start rainbow effect with speed {speed}")
        return f"启动彩虹效果（速度: {speed}x）"

    @function_tool
    def rgb_effect_breathing(self, r: int, g: int, b: int) -> str:
        """
        启动呼吸效果

        Args:
            r: 红色值 (0-255)
            g: 绿色值 (0-255)
            b: 蓝色值 (0-255)

        Returns:
            执行结果描述
        """
        if not all(0 <= val <= 255 for val in [r, g, b]):
            return f"错误：RGB 值必须在 0-255 范围内"

        from lelamp.service import Priority
        self.rgb.dispatch("effect", {"name": "breathing", "color": (r, g, b)}, Priority.NORMAL)
        self.state_manager.set_light_override(duration_s=15.0)
        logger.info(f"Start breathing effect: RGB({r}, {g}, {b})")
        return f"启动呼吸效果: RGB({r}, {g}, {b})"

    @function_tool
    def rgb_effect_random_animation(self) -> str:
        """
        启动随机颜色动画（用于 speaking 状态）

        Returns:
            执行结果描述
        """
        colors = [
            (255, 0, 0),    # 红
            (255, 165, 0),  # 橙
            (255, 255, 0),  # 黄
            (0, 255, 0),    # 绿
            (0, 0, 255),    # 蓝
            (75, 0, 130),   # 靛
            (238, 130, 238) # 紫
        ]
        color = random.choice(colors)

        from lelamp.service import Priority
        self.rgb.dispatch("effect", {"name": "breathing", "color": color}, Priority.HIGH)
        # 注意：speaking 动画不设置 light_override，允许状态切换时自动更新
        logger.info(f"Start random speaking animation: {color}")
        return f"启动随机说话动画"
```

**Step 2: 更新 tools/__init__.py**

```python
"""
Agent 工具模块
"""
from .motor_tools import MotorTools
from .rgb_tools import RGBTools

__all__ = ["MotorTools", "RGBTools"]
```

**Step 3: 编写 rgb_tools 集成测试**

Create: `lelamp/test/integration/test_rgb_tools.py`

```python
"""
RGBTools 集成测试
"""
import pytest
from unittest.mock import Mock
from lelamp.agent.tools.rgb_tools import RGBTools
from lelamp.agent.states import StateManager


@pytest.mark.integration
class TestRGBTools:
    """RGBTools 集成测试套件"""

    @pytest.fixture
    def rgb_tools(self):
        """创建 RGBTools 实例"""
        rgb_service = Mock()
        state_manager = StateManager()
        return RGBTools(rgb_service, state_manager)

    def test_set_rgb_solid_valid(self, rgb_tools):
        """测试设置有效纯色"""
        result = rgb_tools.set_rgb_solid(255, 0, 0)
        assert "255" in result and "0" in result
        rgb_tools.rgb.dispatch.assert_called_once()
        assert rgb_tools.state_manager.is_light_overridden() is True

    def test_set_rgb_solid_invalid(self, rgb_tools):
        """测试无效 RGB 值"""
        result = rgb_tools.set_rgb_solid(300, 0, 0)
        assert "错误" in result
        rgb_tools.rgb.dispatch.assert_not_called()

    def test_paint_rgb_pattern(self, rgb_tools):
        """测试绘制图案"""
        result = rgb_tools.paint_rgb_pattern("heart")
        assert "heart" in result
        rgb_tools.rgb.dispatch.assert_called_once()

    def test_rgb_effect_rainbow(self, rgb_tools):
        """测试彩虹效果"""
        result = rgb_tools.rgb_effect_rainbow(speed=2.0)
        assert "彩虹" in result
        rgb_tools.rgb.dispatch.assert_called_once()

    def test_rgb_effect_breathing(self, rgb_tools):
        """测试呼吸效果"""
        result = rgb_tools.rgb_effect_breathing(0, 140, 255)
        assert "呼吸" in result
        rgb_tools.rgb.dispatch.assert_called_once()

    def test_light_override_blocks_motion(self, rgb_tools):
        """测试灯光覆盖阻止电机动作"""
        # 设置灯光后应该阻止电机
        rgb_tools.set_rgb_solid(255, 255, 255)
        assert rgb_tools.state_manager.is_light_overridden() is True
        # 电机动作会被 StateManager.can_execute_motion() 拒绝
        # （这在 MotorTools 中验证）
```

**Step 4: 运行测试验证**

Run: `uv run pytest lelamp/test/integration/test_rgb_tools.py -v`
Expected: 所有测试通过

**Step 5: Commit**

```bash
git add lelamp/agent/tools/rgb_tools.py lelamp/agent/tools/__init__.py lelamp/test/integration/test_rgb_tools.py
git commit -m "refactor: extract RGB tools from main.py

- Create RGBTools class with solid, pattern, and effect methods
- Implement RGB value validation (0-255)
- Integrate with StateManager for light override
- Add integration tests for all RGB tools"
```

---

### Task 2.4: 创建增强的日志系统

**Files:**
- Create: `lelamp/utils/logging.py`
- Modify: `main.py` (使用新日志系统)

**Step 1: 创建 logging.py**

```python
"""
增强的日志系统 - 支持结构化日志和轮转
"""
import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """结构化 JSON 日志格式"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON"""
        log_data = {
            "timestamp": self.formatTime(record),
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
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_json: bool = True,
) -> None:
    """
    配置日志系统

    Args:
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_dir: 日志目录路径，None 表示不写文件
        enable_json: 是否启用结构化 JSON 日志（文件）
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 控制台日志（人类可读格式）
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    handlers = [console_handler]

    # 文件日志（可选）
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 普通日志文件（人类可读）
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "lelamp.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(console_formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)

        # 结构化 JSON 日志（可选）
        if enable_json:
            json_handler = logging.handlers.RotatingFileHandler(
                log_dir / "lelamp.json.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            json_handler.setFormatter(StructuredFormatter())
            json_handler.setLevel(level)
            handlers.append(json_handler)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有处理器
    root_logger.handlers.clear()

    # 添加新处理器
    for handler in handlers:
        root_logger.addHandler(handler)

    # 抑制第三方库的详细日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("livekit").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称（通常使用 __name__）

    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)
```

**Step 2: 更新 main.py 使用新日志系统**

在 `main.py` 中替换 `_setup_logging()` 函数：

```python
from lelamp.utils.logging import setup_logging
from pathlib import Path
import os

# 替换原来的 _setup_logging() 函数
def _setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    log_dir = Path("logs") if os.getenv("LELAMP_LOG_TO_FILE", "0") == "1" else None
    setup_logging(log_level=log_level, log_dir=log_dir)
```

**Step 3: 编写 logging 单元测试**

Create: `lelamp/test/unit/test_logging.py`

```python
"""
Logging 模块单元测试
"""
import pytest
import logging
import json
import tempfile
from pathlib import Path
from lelamp.utils.logging import setup_logging, get_logger, StructuredFormatter


@pytest.mark.unit
class TestLogging:
    """日志系统测试套件"""

    def test_get_logger(self):
        """测试获取日志记录器"""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_setup_logging_console_only(self):
        """测试仅控制台日志"""
        setup_logging(log_level="INFO", log_dir=None)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)

    def test_setup_logging_with_file(self):
        """测试文件日志"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_level="DEBUG", log_dir=log_dir, enable_json=True)

            # 写入测试日志
            logger = get_logger("test")
            logger.info("Test message")

            # 验证日志文件存在
            assert (log_dir / "lelamp.log").exists()
            assert (log_dir / "lelamp.json.log").exists()

    def test_structured_formatter(self):
        """测试结构化 JSON 格式化器"""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_log_rotation(self):
        """测试日志轮转配置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_level="INFO", log_dir=log_dir)

            # 获取 RotatingFileHandler
            root_logger = logging.getLogger()
            rotating_handlers = [
                h for h in root_logger.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]

            assert len(rotating_handlers) > 0
            handler = rotating_handlers[0]
            assert handler.maxBytes == 10 * 1024 * 1024  # 10 MB
            assert handler.backupCount == 5
```

**Step 4: 运行测试验证**

Run: `uv run pytest lelamp/test/unit/test_logging.py -v`
Expected: 所有测试通过

**Step 5: Commit**

```bash
git add lelamp/utils/logging.py lelamp/test/unit/test_logging.py main.py
git commit -m "feat: add enhanced logging system with rotation and JSON support

- Create structured JSON formatter for machine-readable logs
- Implement rotating file handlers (10MB, 5 backups)
- Support both human-readable and JSON log formats
- Add comprehensive unit tests
- Update main.py to use new logging system"
```

---

由于实施计划非常长，让我继续创建剩余的阶段（Task 2.5-2.7, Phase 3, Phase 4）...

**Step 6: 继续编写实施计划（剩余任务）**

```markdown
### Task 2.5: 创建 vision_tools 和 system_tools

（省略详细步骤，结构类似 Task 2.2 和 2.3）

**Summary:**
- Extract vision_answer, check_homework, capture_to_feishu to `vision_tools.py`
- Extract set_volume, web_search, tune_motor_pid to `system_tools.py`
- Write integration tests

### Task 2.6: 创建 lelamp/agent/lelamp_agent.py

**Files:**
- Create: `lelamp/agent/lelamp_agent.py`
- Update: `lelamp/agent/__init__.py`

**Summary:**
- Extract LeLamp class from main.py
- Rename to LeLampAgent
- Initialize services and tools
- Implement entrypoint and state callbacks
- Write integration tests

### Task 2.7: 简化 main.py 为入口文件

**Files:**
- Modify: `main.py`
- Create: `main_legacy.py` (backup)

**Summary:**
- Backup current main.py to main_legacy.py
- Rewrite main.py to ~50 lines
- Import LeLampAgent from lelamp.agent
- Keep only entrypoint logic
- Test end-to-end

---

## Phase 3: 前端现代化（第三阶段）

### Task 3.1: 初始化 Vue 3 + Vite 项目

### Task 3.2: 创建 Pinia Store

### Task 3.3: 实现 VideoPanel 组件

### Task 3.4: 实现 ControlPanel 和子组件

### Task 3.5: 集成 LiveKit Client

### Task 3.6: 构建和部署测试

---

## Phase 4: CI/CD 集成（第四阶段）

### Task 4.1: 创建 GitHub Actions - Tests

### Task 4.2: 创建 GitHub Actions - Lint

### Task 4.3: 创建 GitHub Actions - Frontend

### Task 4.4: 配置 Codecov

### Task 4.5: 添加 README badges

---

## 验收标准

完成所有任务后，验证：

1. ✅ `uv run pytest -m unit` 全部通过
2. ✅ `uv run pytest -m integration` 全部通过
3. ✅ `uv run ruff check .` 无错误
4. ✅ `main.py` 代码行数 < 100
5. ✅ 测试覆盖率 > 60%
6. ✅ 前端 `npm run build` 成功
7. ✅ CI/CD 工作流运行成功

---

## 回滚计划

如遇到问题：

1. **Phase 1-2**: `git revert` 相关 commits
2. **Phase 2**: 使用 `main_legacy.py` 恢复
3. **Phase 3**: 使用旧 `web_client` 目录
4. **Phase 4**: 禁用 GitHub Actions

---

**计划创建时间**: 2026-03-17
**预计完成时间**: 5 个工作日
```

现在让我保存完整的实施计划（由于篇幅限制，我会创建一个简化但完整的版本）：