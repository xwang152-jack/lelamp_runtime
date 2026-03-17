# LeLamp Runtime 项目优化设计文档

**日期**: 2026-03-17
**版本**: 1.0
**状态**: 已批准

## 1. 概述

本设计文档基于 `IMPROVEMENT_REPORT.md` 的评估结果，提出针对 LeLamp Runtime 项目的短期和中期优化方案。优化范围包括：测试体系完善、代码重构、日志增强、前端现代化和 CI/CD 集成。

### 1.1 优化目标

- **提升代码可维护性**：将 1343 行的 `main.py` 重构为模块化架构
- **增强测试覆盖**：建立完整的单元测试、集成测试和硬件测试体系
- **改善用户体验**：使用 Vue 3 重构前端，提升交互质量
- **自动化质量保证**：通过 CI/CD 确保代码质量和测试通过

### 1.2 预期成果

- main.py 从 1343 行减少到 ~50 行
- 测试覆盖率达到 60%+（核心模块 80%+）
- 前端代码可维护性提升 3 倍
- 每次 PR 自动执行测试和代码检查

## 2. 整体架构

### 2.1 新项目结构

```
lelamp_runtime/
├── lelamp/
│   ├── agent/                    # 🆕 Agent 核心逻辑（从 main.py 迁移）
│   │   ├── __init__.py          # 导出 LeLampAgent
│   │   ├── lelamp_agent.py      # Agent 类定义
│   │   ├── tools/               # 🆕 Function tools 模块
│   │   │   ├── __init__.py
│   │   │   ├── motor_tools.py   # 电机控制工具
│   │   │   ├── rgb_tools.py     # RGB 灯光工具
│   │   │   ├── vision_tools.py  # 视觉工具
│   │   │   └── system_tools.py  # 系统工具
│   │   └── states.py            # 🆕 会话状态管理
│   ├── service/                  # 保持不变
│   ├── integrations/             # 保持不变
│   ├── utils/                    # 保持不变
│   └── test/                     # 重组测试结构
│       ├── unit/                 # 🆕 单元测试
│       │   ├── __init__.py
│       │   ├── test_config.py
│       │   ├── test_rate_limiter.py
│       │   ├── test_cache.py
│       │   └── test_url_validation.py  # 从根目录迁移
│       ├── integration/          # 🆕 集成测试
│       │   ├── __init__.py
│       │   ├── test_agent_tools.py
│       │   └── test_service_integration.py
│       └── hardware/             # 🆕 硬件测试
│           ├── __init__.py
│           ├── test_motors.py    # 从根目录迁移
│           ├── test_rgb.py       # 从根目录迁移
│           └── test_audio.py     # 从根目录迁移
├── web_client/                   # 🔄 重构为 Vue 3 + Vite
│   ├── public/                   # 静态资源
│   ├── src/
│   │   ├── components/           # Vue 组件
│   │   │   ├── VideoPanel.vue
│   │   │   ├── ControlPanel.vue
│   │   │   ├── LightControl.vue
│   │   │   ├── MotorControl.vue
│   │   │   └── VisionControl.vue
│   │   ├── stores/               # Pinia 状态管理
│   │   │   └── livekit.ts
│   │   ├── App.vue
│   │   └── main.ts
│   ├── index.html
│   ├── package.json              # 前端依赖
│   ├── tsconfig.json
│   └── vite.config.ts            # 构建配置
├── tests/                        # 🆕 项目级测试配置
│   └── conftest.py               # pytest 共享配置
├── .github/
│   └── workflows/
│       ├── test.yml              # 🆕 自动化测试
│       └── lint.yml              # 🆕 代码检查
├── main.py                       # 🔄 简化为入口文件（~50 行）
├── pyproject.toml                # 🔄 增加开发依赖和 pytest 配置
├── pytest.ini                    # 🆕 pytest 配置文件
└── .github/                      # 🆕 CI/CD 配置
```

### 2.2 核心变更说明

#### 2.2.1 main.py 瘦身

**变更前**（1343 行）：
- Agent 类定义
- 所有 @function_tool 方法
- 会话状态管理
- 配置加载
- 服务初始化

**变更后**（~50 行）：
```python
# main.py - 入口文件
from lelamp.agent import LeLampAgent
from lelamp.config import load_config
from livekit.agents import cli, WorkerOptions

def main():
    config = load_config()
    agent = LeLampAgent(config)
    cli.run_app(WorkerOptions(entrypoint=agent.entrypoint))

if __name__ == "__main__":
    main()
```

#### 2.2.2 Agent 模块化

**lelamp/agent/lelamp_agent.py**：
- 包含 `LeLampAgent` 类定义
- 会话生命周期管理
- 状态回调处理
- 工具注册

**lelamp/agent/tools/**：
- 按功能拆分 function_tool 到不同文件
- `motor_tools.py`: play_recording, move_joint, get_joint_positions 等
- `rgb_tools.py`: set_rgb_solid, paint_rgb_pattern, rgb_effect_* 等
- `vision_tools.py`: vision_answer, check_homework, capture_to_feishu
- `system_tools.py`: set_volume, web_search, get_motor_health 等

**lelamp/agent/states.py**：
- 会话状态枚举和管理
- Cooldown 和 override 逻辑

## 3. 实施阶段

### 3.1 第一阶段：测试基础设施（1 天）

#### 3.1.1 pytest 配置

**pytest.ini**：
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

**pyproject.toml 增加**：
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

#### 3.1.2 新增单元测试

**tests/conftest.py**：
```python
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
        # ... 其他配置
    )

@pytest.fixture
def mock_motors_service():
    """Mock MotorsService"""
    return Mock()
```

**lelamp/test/unit/test_rate_limiter.py**：
```python
import pytest
import time
from lelamp.utils.rate_limiter import RateLimiter, get_rate_limiter

@pytest.mark.unit
def test_rate_limiter_basic():
    limiter = RateLimiter(rate=2.0, capacity=2)  # 2 req/s
    assert limiter.try_acquire() == True
    assert limiter.try_acquire() == True
    assert limiter.try_acquire() == False  # 超出容量

@pytest.mark.unit
def test_rate_limiter_refill():
    limiter = RateLimiter(rate=1.0, capacity=1)
    assert limiter.try_acquire() == True
    time.sleep(1.1)
    assert limiter.try_acquire() == True  # 令牌已补充
```

**lelamp/test/unit/test_cache.py**：
```python
import pytest
import time
from lelamp.cache.cache_manager import VisionCache, SearchCache

@pytest.mark.unit
def test_vision_cache_basic():
    cache = VisionCache()
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None

@pytest.mark.unit
def test_cache_ttl_expiration():
    cache = VisionCache(default_ttl=0.5)  # 0.5秒过期
    cache.set("key1", "value1")
    time.sleep(0.6)
    assert cache.get("key1") is None  # 已过期
```

#### 3.1.3 集成测试

**lelamp/test/integration/test_agent_tools.py**：
```python
import pytest
from unittest.mock import Mock, patch
from lelamp.agent.tools.motor_tools import MotorTools

@pytest.mark.integration
def test_play_recording_with_service(mock_motors_service):
    tools = MotorTools(mock_motors_service)
    result = tools.play_recording("wave")
    mock_motors_service.dispatch.assert_called_once()
    assert "播放" in result
```

### 3.2 第二阶段：代码重构（1 天）

#### 3.2.1 创建 lelamp/agent 模块

**步骤**：
1. 创建 `lelamp/agent/` 目录结构
2. 从 `main.py` 提取 `LeLamp` 类到 `lelamp_agent.py`
3. 重命名为 `LeLampAgent`
4. 提取所有 `@function_tool` 方法到 `tools/` 子模块
5. 提取状态管理逻辑到 `states.py`

**lelamp/agent/__init__.py**：
```python
from .lelamp_agent import LeLampAgent

__all__ = ["LeLampAgent"]
```

**lelamp/agent/lelamp_agent.py** 结构：
```python
class LeLampAgent:
    def __init__(self, config: AppConfig):
        self.config = config
        self._initialize_services()
        self._register_tools()

    def _initialize_services(self):
        # 初始化 MotorsService, RGBService, VisionService
        pass

    def _register_tools(self):
        # 注册所有 function tools
        pass

    async def entrypoint(self, ctx: JobContext):
        # LiveKit agent 入口
        pass
```

**lelamp/agent/tools/motor_tools.py**：
```python
from livekit.agents import function_tool

class MotorTools:
    def __init__(self, motors_service):
        self.motors = motors_service

    @function_tool
    def play_recording(self, name: str) -> str:
        """播放预录制的电机动作"""
        # 实现逻辑
        pass

    @function_tool
    def move_joint(self, joint: str, position: float) -> str:
        """移动单个关节"""
        # 实现逻辑
        pass
```

#### 3.2.2 增强日志系统

**lelamp/utils/logging.py**（新建）：
```python
import logging
import logging.handlers
import json
from pathlib import Path

class StructuredFormatter(logging.Formatter):
    """结构化 JSON 日志格式"""
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logging(log_level: str = "INFO", log_dir: Path = None):
    """配置日志系统"""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 控制台日志（人类可读）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    # 文件日志（结构化 JSON + 轮转）
    if log_dir:
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "lelamp.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setFormatter(StructuredFormatter())

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    else:
        logging.basicConfig(level=level, handlers=[console_handler])
```

**main.py 更新**：
```python
from lelamp.utils.logging import setup_logging
from pathlib import Path

setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=Path("logs") if os.getenv("LELAMP_LOG_TO_FILE") else None
)
```

#### 3.2.3 重组测试目录

**迁移现有测试**：
```bash
# 移动硬件测试
mv lelamp/test/test_motors.py lelamp/test/hardware/
mv lelamp/test/test_rgb.py lelamp/test/hardware/
mv lelamp/test/test_audio.py lelamp/test/hardware/

# 移动单元测试
mv lelamp/test/test_security.py lelamp/test/unit/
mv lelamp/test/test_url_validation.py lelamp/test/unit/
mv lelamp/test/test_ota.py lelamp/test/unit/
```

### 3.3 第三阶段：前端现代化（2 天）

#### 3.3.1 技术栈选择

- **框架**: Vue 3（Composition API）
- **构建工具**: Vite
- **语言**: TypeScript
- **状态管理**: Pinia
- **UI 库**: 原生 CSS + Tailwind CSS（可选）
- **LiveKit SDK**: `livekit-client`

#### 3.3.2 项目初始化

```bash
cd web_client
npm create vite@latest . -- --template vue-ts
npm install livekit-client pinia
```

#### 3.3.3 组件结构

**src/App.vue**：
```vue
<template>
  <div class="app">
    <VideoPanel />
    <ControlPanel />
  </div>
</template>
```

**src/components/VideoPanel.vue**：
- 视频流显示
- 音频可视化

**src/components/ControlPanel.vue**：
```vue
<template>
  <div class="control-panel">
    <LightControl />
    <MotorControl />
    <VisionControl />
  </div>
</template>
```

**src/stores/livekit.ts**：
```typescript
import { defineStore } from 'pinia'
import { Room, RoomEvent } from 'livekit-client'

export const useLiveKitStore = defineStore('livekit', {
  state: () => ({
    room: null as Room | null,
    connected: false,
    participants: [],
  }),
  actions: {
    async connect(url: string, token: string) {
      this.room = new Room()
      await this.room.connect(url, token)
      this.connected = true
    },
    async disconnect() {
      await this.room?.disconnect()
      this.connected = false
    }
  }
})
```

#### 3.3.4 保留功能映射

| 原功能 | 新实现 |
|--------|--------|
| 视频显示 | VideoPanel.vue |
| 灯光控制 | LightControl.vue |
| 电机控制 | MotorControl.vue |
| 视觉捕获 | VisionControl.vue |
| 音量控制 | 集成到 ControlPanel |

#### 3.3.5 构建配置

**vite.config.ts**：
```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: '../dist/web_client',
    emptyOutDir: true
  }
})
```

### 3.4 第四阶段：CI/CD 集成（1 天）

#### 3.4.1 GitHub Actions - 测试工作流

**.github/workflows/test.yml**：
```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install UV
      run: curl -LsSf https://astral.sh/uv/install.sh | sh

    - name: Install dependencies
      run: |
        uv sync --extra dev

    - name: Run unit tests
      run: |
        uv run pytest -m unit --cov=lelamp --cov-report=xml

    - name: Run integration tests
      run: |
        uv run pytest -m integration

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
```

#### 3.4.2 GitHub Actions - Lint 工作流

**.github/workflows/lint.yml**：
```yaml
name: Lint

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - name: Install UV
      run: curl -LsSf https://astral.sh/uv/install.sh | sh

    - name: Install dependencies
      run: uv sync --extra dev

    - name: Run Ruff
      run: |
        uv run ruff check .
        uv run ruff format --check .
```

#### 3.4.3 前端 CI

**.github/workflows/frontend.yml**：
```yaml
name: Frontend

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'

    - name: Install dependencies
      working-directory: ./web_client
      run: npm ci

    - name: Lint
      working-directory: ./web_client
      run: npm run lint

    - name: Build
      working-directory: ./web_client
      run: npm run build
```

## 4. 测试策略

### 4.1 测试分层

#### 4.1.1 单元测试（Unit Tests）
- **范围**: utils, cache, integrations（纯逻辑部分）
- **特点**: 快速、无外部依赖、可并行
- **Mock**: 所有外部服务和硬件
- **目标覆盖率**: 80%+

#### 4.1.2 集成测试（Integration Tests）
- **范围**: agent tools, service interactions
- **特点**: 使用 Mock，测试模块间协作
- **目标覆盖率**: 60%+

#### 4.1.3 硬件测试（Hardware Tests）
- **范围**: 电机、RGB、音频
- **特点**: 需要实际硬件
- **运行**: 手动或在硬件 CI 环境

### 4.2 测试覆盖目标

| 模块 | 目标覆盖率 |
|------|-----------|
| lelamp/utils/ | 85% |
| lelamp/cache/ | 90% |
| lelamp/integrations/ | 70% |
| lelamp/agent/ | 60% |
| lelamp/service/ | 50% |
| 整体 | 60% |

## 5. 质量保证

### 5.1 代码风格

- **Linter**: Ruff
- **Formatter**: Ruff format
- **配置**: 继承现有规则，增加严格模式

### 5.2 提交规范

- 遵循 Conventional Commits
- 示例: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

### 5.3 PR 流程

1. 创建功能分支
2. 编写代码 + 测试
3. 本地运行 `pytest` 和 `ruff`
4. 提交 PR
5. 等待 CI 通过
6. Code Review
7. 合并到 main

## 6. 风险与缓解

### 6.1 风险点

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 重构引入 bug | 高 | 中 | 先写测试，渐进式重构 |
| 前端迁移工作量大 | 中 | 高 | 分阶段实施，保留旧版本 |
| CI/CD 配置错误 | 低 | 低 | 在测试分支验证 |
| 硬件测试阻塞 CI | 中 | 中 | 将硬件测试标记为 `@pytest.mark.hardware`，CI 不运行 |

### 6.2 回滚策略

- 每个阶段独立分支，可随时回滚
- 保留原 `main.py` 为 `main_legacy.py` 备份
- 前端保留旧版本在 `web_client_legacy/`

## 7. 时间表

| 阶段 | 工作日 | 主要任务 |
|------|--------|----------|
| 第一阶段 | 1 天 | pytest 配置、单元测试、集成测试 |
| 第二阶段 | 1 天 | main.py 重构、agent 模块化、日志增强 |
| 第三阶段 | 2 天 | Vue 3 前端重构、组件化 |
| 第四阶段 | 1 天 | CI/CD 配置、工作流验证 |
| **总计** | **5 天** | - |

## 8. 成功标准

### 8.1 代码质量
- ✅ main.py 代码行数 < 100
- ✅ 整体测试覆盖率 > 60%
- ✅ Ruff lint 无错误
- ✅ 所有单元测试通过

### 8.2 架构改善
- ✅ Agent 逻辑独立到 `lelamp/agent/` 包
- ✅ Function tools 按功能分类到子模块
- ✅ 测试分层（unit/integration/hardware）

### 8.3 用户体验
- ✅ 前端使用 Vue 3 + TypeScript
- ✅ 保留所有原有功能
- ✅ 构建时间 < 10 秒

### 8.4 自动化
- ✅ PR 自动运行测试
- ✅ Lint 检查自动化
- ✅ 前端自动构建验证

## 9. 后续计划

优化完成后的后续工作：

1. **性能优化** - 分析瓶颈，优化响应时间
2. **文档更新** - 更新 README 和 API 文档
3. **监控告警** - 集成 Sentry 或其他监控工具
4. **长期规划** - 插件系统、边缘计算优化（来自 IMPROVEMENT_REPORT.md）

## 10. 总结

本设计文档提出了一个渐进式、低风险的项目优化方案。通过测试先行、模块化重构、前端现代化和 CI/CD 集成，将显著提升 LeLamp Runtime 的代码质量、可维护性和开发效率。

---

**文档版本历史**:
- v1.0 (2026-03-17): 初始版本，已批准
