# LeLamp Runtime 后端开发计划

**日期**: 2026-03-17
**版本**: 1.0
**状态**: 规划中

---

## 1. 项目现状分析

### 1.1 当前架构

**后端技术栈**:
- Python 3.12+
- LiveKit Agents SDK（实时语音和视频）
- DeepSeek LLM（对话 AI）
- Qwen VL（视觉理解）
- Baidu Speech（语音识别和合成）
- Feetech Servo SDK（电机控制）
- rpi-ws281x（RGB LED 控制）

**当前代码状态**:
- ✅ Phase 1-2 完成：测试基础设施 + 代码重构
- ⏸️ Phase 2 待完成：后端模块化重构（Task 2.5-2.7）
- ⏸️ Phase 3 待开始：前端现代化（已完成）
- ⏸️ Phase 4 待开始：CI/CD 集成

**主要问题**:
1. `main.py` 文件过大（241 行），职责过多
2. 所有功能耦合在一个文件中，难以维护
3. 工具方法未模块化，缺少清晰的边界
4. 缺少单元测试和集成测试

---

## 2. 后端开发路线图

### Phase 2：后端模块化重构 ⏳

**目标**: 将 `main.py` 重构为模块化架构

**任务清单**:

#### Task 2.5: 提取工具类
- [ ] **2.5.1**: 创建 `vision_tools.py`
  - 从 main.py 提取视觉工具方法
  - 实现 `VisionTools` 类
  - 添加集成测试

- [ ] **2.5.2**: 创建 `system_tools.py`
  - 从 main.py 提取系统工具方法
  - 实现 `SystemTools` 类
  - 添加集成测试

- [ ] **2.5.3**: 更新 `tools/__init__.py`
  - 导出所有工具类
  - 统一工具接口

#### Task 2.6: 创建核心代理类
- [ ] 创建 `lelamp_agent.py`
  - 提取 `LeLamp` 类
  - 集成所有工具类
  - 实现 Data Channel 处理
  - 添加集成测试

#### Task 2.7: 简化入口文件
- [ ] 重构 `main.py`
  - 简化为纯入口文件（< 100 行）
  - 提取配置加载函数
  - 保持功能完整性
  - 验证所有功能正常

**预计工时**: 4-6 小时

---

### Phase 3：后端功能增强 📋

**目标**: 添加新功能和优化

#### 3.1 API 服务器

**功能**:
- RESTful API 接口
- 设备状态查询 API
- 历史记录 API
- 健康监控 API
- WebSocket 实时推送

**技术选型**:
- FastAPI（现代、快速、自动文档）
- Pydantic（数据验证）
- WebSocket（实时通信）

**实施步骤**:
1. 创建 `api/` 目录
2. 实现 FastAPI 应用
3. 定义 API 路由
4. 集成现有服务
5. 添加 API 文档

#### 3.2 数据持久化

**功能**:
- 对话历史记录存储
- 操作日志记录
- 设备状态记录
- 用户偏好设置

**技术选型**:
- SQLite（本地轻量级）
- 或 PostgreSQL（生产环境）
- SQLAlchemy ORM

**数据模型**:
```python
# Conversation
- id, timestamp, messages, duration

# OperationLog
- id, timestamp, type, action, params, success

# DeviceState
- id, timestamp, motor_positions, health_status, light_color

# UserSettings
- id, theme, language, notifications
```

#### 3.3 实时数据分析

**功能**:
- 使用统计（每日/每周/每月）
- 对话分析（关键词、情感）
- 设备健康趋势
- 异常检测和告警

**技术选型**:
- Prometheus（指标收集）
- Grafana（可视化）
- 自定义分析脚本

#### 3.4 多用户支持

**功能**:
- 用户认证和授权
- 多设备管理
- 用户权限控制
- 使用配额限制

**技术选型**:
- JWT Token 认证
- RBAC 权限模型
- 设备绑定机制

---

### Phase 4：后端优化与部署 🚀

#### 4.1 性能优化

**目标**:
- 减少响应延迟
- 提高并发处理能力
- 优化资源使用

**优化项**:
- 异步 I/O 优化
- 缓存策略改进
- 数据库查询优化
- 内存使用优化

#### 4.2 安全加固

**目标**:
- 提升系统安全性
- 保护用户隐私
- 防止恶意攻击

**安全项**:
- API 速率限制
- 输入验证加强
- SQL 注入防护
- XSS 防护
- HTTPS 强制

#### 4.3 CI/CD 集成

**目标**:
- 自动化测试和部署
- 代码质量保障
- 快速迭代发布

**CI/CD 流程**:
- GitHub Actions 工作流
- 自动化测试（pytest）
- 代码覆盖率检查
- 自动构建和部署
- 版本标签管理

#### 4.4 监控和运维

**目标**:
- 实时监控服务状态
- 快速发现和定位问题
- 自动化运维操作

**监控项**:
- 应用性能监控（APM）
- 错误日志聚合
- 资源使用监控
- 告警通知机制

---

## 3. 详细实施计划

### 立即可做：Phase 2 重构

#### Task 2.5.1: 创建 vision_tools.py

**文件**: `lelamp/agent/tools/vision_tools.py`

**必需方法**:
```python
class VisionTools:
    @function_tool
    async def vision_answer(self, question: str) -> str:
        """拍照识别并回答问题"""

    @function_tool
    async def check_homework(self) -> str:
        """检查作业"""

    @function_tool
    async def capture_to_feishu(self) -> str:
        """拍照并推送到飞书"""
```

**关键点**:
- 使用 `StateManager` 管理灯光状态
- 实现速率限制
- 处理隐私保护
- 线程安全设计

**测试**: `lelamp/test/integration/test_vision_tools.py`
- 8-10 个集成测试
- Mock 所有外部依赖

#### Task 2.5.2: 创建 system_tools.py

**文件**: `lelamp/agent/tools/system_tools.py`

**必需方法**（按优先级）:
```python
class SystemTools:
    # 高优先级
    @function_tool
    async def get_available_recordings(self) -> str:
        """获取可用的录制动画列表"""

    @function_tool
    async def set_rgb_brightness(self, percent: int) -> str:
        """设置灯光亮度"""

    @function_tool
    async def set_volume(self, volume_percent: int) -> str:
        """设置系统音量"""

    @function_tool
    async def web_search(self, query: str) -> str:
        """联网搜索"""

    # 中优先级
    @function_tool
    async def rgb_effect_wave(self, ...) -> str:
        """波纹效果"""

    @function_tool
    async def rgb_effect_fire(self, ...) -> str:
        """火焰效果"""

    @function_tool
    async def rgb_effect_emoji(self, ...) -> str:
        """表情动画"""

    @function_tool
    async def check_for_updates(self) -> str:
        """检查更新"""

    @function_tool
    async def perform_ota_update(self) -> str:
        """执行 OTA 更新"""

    # 低优先级
    @function_tool
    async def tune_motor_pid(self, ...) -> str:
        """电机 PID 调参"""

    @function_tool
    async def reset_motor_health_stats(self, motor_name: str) -> str:
        """重置健康统计"""

    @function_tool
    async def get_rate_limit_stats(self) -> str:
        """获取速率限制统计"""
```

**测试**: `lelamp/test/integration/test_system_tools.py`
- 15-20 个集成测试
- 覆盖所有主要功能组

#### Task 2.6: 创建 lelamp_agent.py

**文件**: `lelamp/agent/lelamp_agent.py`

**核心职责**:
- 继承 `livekit.agents.Agent`
- 集成所有工具类
- 管理会话状态
- 处理 Data Channel 消息

**关键方法**:
```python
class LeLamp(Agent):
    def __init__(self, port, lamp_id, vision_service,
                 qwen_client, ota_url, motor_config):
        # 初始化服务
        # 创建工具实例
        # 设置状态管理器

    async def note_user_text(self, text: str):
        """记录用户输入"""

    async def set_conversation_state(self, state: str):
        """设置会话状态并更新灯光"""

    async def handle_data_message(self, data: bytes, participant):
        """处理 Web Client 发送的 Data Channel 消息"""
```

**测试**: `lelamp/test/integration/test_lelamp_agent.py`
- 测试状态切换
- 测试 Data Channel 消息处理
- Mock 所有服务

#### Task 2.7: 简化 main.py

**目标**: 将 `main.py` 从 241 行简化到 < 100 行

**保留内容**:
- 配置加载
- 日志设置
- 代理创建和启动
- 会话管理

**提取到独立模块**:
- `_load_config()` → `lelamp/config.py`
- `_setup_logging()` → `lelamp/utils/logging.py`
- `_build_vad()` → 保留或提取
- `LeLamp` 类 → `lelamp/agent/lelamp_agent.py`

---

### 中期规划：Phase 3 功能增强

#### 3.1 FastAPI 服务器

**文件结构**:
```
lelamp/
├── api/
│   ├── __init__.py
│   ├── app.py          # FastAPI 应用
│   ├── routes/
│   │   ├── devices.py  # 设备相关 API
│   │   ├── history.py  # 历史记录 API
│   │   └── health.py   # 健康监控 API
│   ├── models/
│   │   ├── conversation.py
│   │   ├── operation.py
│   │   └── device_state.py
│   └── dependencies.py
```

**API 端点示例**:
```python
# 设备状态
GET    /api/devices/{lamp_id}/state
POST   /api/devices/{lamp_id}/command

# 历史记录
GET    /api/conversations
GET    /api/conversations/{id}
GET    /api/operations

# 健康监控
GET    /api/health/{lamp_id}
GET    /api/health/{lamp_id}/history
```

#### 3.2 数据持久化

**文件结构**:
```
lelamp/
├── database/
│   ├── __init__.py
│   ├── base.py        # SQLAlchemy Base
│   ├── models.py      # ORM 模型
│   ├── crud.py        # CRUD 操作
│   └── session.py     # 数据库会话
```

**实施步骤**:
1. 定义 ORM 模型
2. 创建数据库初始化脚本
3. 实现 CRUD 操作
4. 集成到现有服务
5. 数据迁移脚本

---

## 4. 开发优先级建议

### 方案 A：完成重构优先（推荐）

**理由**:
- 模块化是后续开发的基础
- 提高代码可维护性
- 便于添加新功能

**执行顺序**:
1. Phase 2.5-2.7（模块化重构）
2. Phase 3.1（FastAPI 服务器）
3. Phase 3.2（数据持久化）
4. Phase 4（优化和部署）

### 方案 B：功能优先

**理由**:
- 快速交付可见功能
- 满足用户需求
- 后续优化架构

**执行顺序**:
1. Phase 3.1（FastAPI 服务器）
2. Phase 3.2（数据持久化）
3. Phase 2.5-2.7（模块化重构）
4. Phase 4（优化和部署）

### 方案 C：并行开发

**理由**:
- 前后端同步开发
- 快速推进整体进度

**执行顺序**:
1. Phase 2（后端重构）+ Phase 1（前端）并行
2. Phase 3（后端功能）+ Phase 2（前端功能）并行
3. Phase 4（集成测试和优化）

---

## 5. 技术债务清单

### 当前技术债务

| 债务项 | 严重性 | 影响 | 建议 |
|--------|--------|------|------|
| main.py 过大（241 行） | 高 | 维护困难 | 立即重构 |
| 缺少单元测试 | 高 | 质量风险 | 补充测试 |
| 工具方法未模块化 | 中 | 扩展困难 | 模块化 |
| 缺少 API 文档 | 中 | 接口不清晰 | 添加文档 |
| 缺少数据持久化 | 中 | 数据丢失 | 实现存储 |
| 错误处理不统一 | 低 | 稳定性 | 统一处理 |

### 偿还计划

**高优先级**（立即处理）:
1. 重构 main.py（Phase 2.5-2.7）
2. 添加单元测试（提高覆盖率）

**中优先级**（1-2 个月内）:
3. 实现 API 服务器（Phase 3.1）
4. 数据持久化（Phase 3.2）

**低优先级**（3 个月内）:
5. 统一错误处理
6. 完善 API 文档
7. 性能优化

---

## 6. 测试策略

### 单元测试

**目标覆盖率**: 80%+

**测试框架**:
- pytest（测试运行器）
- pytest-asyncio（异步测试）
- pytest-mock（Mock 对象）

**测试文件结构**:
```
lelamp/test/
├── unit/
│   ├── test_config.py
│   ├── test_states.py
│   ├── test_cache.py
│   └── test_rate_limiter.py
├── integration/
│   ├── test_motor_tools.py
│   ├── test_rgb_tools.py
│   ├── test_vision_tools.py
│   ├── test_system_tools.py
│   └── test_lelamp_agent.py
└── hardware/
    ├── test_motors.py
    ├── test_rgb.py
    └── test_audio.py
```

### 集成测试

**测试场景**:
1. 完整对话流程
2. 视觉问答流程
3. 电机控制流程
4. 灯光控制流程
5. OTA 更新流程

### 端到端测试

**测试工具**:
- Playwright（浏览器自动化）
- 硬件测试台（实际设备）

---

## 7. 部署策略

### 开发环境

```bash
# 启动开发服务器
uv run main.py console

# 或使用 FastAPI（开发模式）
uv run uvicorn lelamp.api.app:app --reload
```

### 生产环境

**部署方式**:
1. **Raspberry Pi 部署**
   - 系统服务（systemd）
   - 自动启动
   - 日志轮转

2. **Docker 容器化**
   - Dockerfile 编写
   - 镜像构建
   - 容器编排

3. **云服务部署**
   - AWS/GCP/Azure
   - 负载均衡
   - 自动扩缩容

---

## 8. 监控和运维

### 日志管理

**日志级别**:
- DEBUG: 开发调试
- INFO: 正常运行
- WARNING: 警告信息
- ERROR: 错误信息
- CRITICAL: 严重错误

**日志轮转**:
- 10MB 单文件大小限制
- 保留 5 个备份
- 自动压缩旧日志

### 性能监控

**关键指标**:
- 响应时间（P50, P95, P99）
- 并发连接数
- CPU/内存使用率
- 错误率

**监控工具**:
- Prometheus（指标收集）
- Grafana（可视化）
- Sentry（错误追踪）

---

## 9. 下一步行动

### 立即可做

1. **完成 Phase 2 重构**
   - 开始 Task 2.5.1（vision_tools.py）
   - 预计工时：2-3 小时

2. **补充单元测试**
   - 为现有模块添加测试
   - 目标覆盖率：80%+

3. **完善文档**
   - API 接口文档
   - 架构设计文档
   - 部署运维文档

### 本周计划

- [ ] 完成 Task 2.5（工具类提取）
- [ ] 完成 Task 2.6（代理类创建）
- [ ] 完成 Task 2.7（简化 main.py）
- [ ] 验证所有功能正常

### 本月计划

- [ ] Phase 3.1：FastAPI 服务器
- [ ] Phase 3.2：数据持久化
- [ ] 补充集成测试
- [ ] 性能基准测试

---

## 10. 总结

**当前状态**:
- ✅ Phase 1（测试基础设施）完成
- ⏳ Phase 2（模块化重构）待完成
- ⏸️ Phase 3（功能增强）待规划
- ⏸️ Phase 4（优化部署）待规划

**建议**:
优先完成 Phase 2 重构，为后续开发打下坚实基础。

---

**文档版本**: 1.0
**创建日期**: 2026-03-17
**最后更新**: 2026-03-17
