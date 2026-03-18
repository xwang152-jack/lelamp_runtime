# 优化执行会话指南

## 📋 会话设置

您现在位于一个独立的 **git worktree** 中，专门用于执行全面优化计划。

**Worktree 路径**: `/Users/jackwang/lelamp_runtime/.claude/worktrees/optimization-implementation`
**当前分支**: `optimization-implementation`
**基础分支**: `main` (commit 32d3263)

---

## 🎯 执行计划

详细的实施计划位于: `docs/plans/2025-03-19-comprehensive-optimization-plan.md`

### Phase 1: 安全基础设施 (当前阶段)

**任务列表:**
- ✅ Task 1: 创建用户认证基础模型
- ✅ Task 2: 实现 JWT 认证服务
- ✅ Task 3: 创建认证 API 路由
- ✅ Task 4: 创建认证中间件
- ✅ Task 5: WebSocket 认证集成

---

## 🚀 如何在新会话中开始执行

### 步骤 1: 启动新的 Claude Code 会话

```bash
cd /Users/jackwang/lelamp_runtime/.claude/worktrees/optimization-implementation
```

然后在新的 Claude Code 会话中使用以下指令：

### 步骤 2: 告诉新会话执行计划

**复制以下指令到新会话:**

```
我将执行 LeLamp Runtime 的全面优化计划。

请使用 superpowers:executing-plans 技能来执行以下实施计划：
docs/plans/2025-03-19-comprehensive-optimization-plan.md

这是一个渐进式三阶段优化计划：
1. Phase 1: 安全基础设施 (用户认证、JWT、API 安全)
2. Phase 2: 功能完善 (用户管理界面、视觉功能、设备管理)
3. Phase 3: 性能优化 (数据库、API、前端性能)

请从 Phase 1 的 Task 1 开始，按顺序执行每个任务。
每个任务都包含详细的代码示例、测试步骤和提交指令。

重要提示：
- 遵循 TDD 原则：先写测试，再实现功能
- 每个任务完成后提交代码
- 运行所有测试确保质量
- 遇到问题及时报告

开始执行吧！
```

---

## 📊 执行跟踪

### 已完成

- 📝 2025-03-19: 创建实施计划文档

### 进行中

- 🔄 等待在新会话中开始执行

### 待执行

- Phase 1: Task 1-5
- Phase 2: Task 6-24
- Phase 3: Task 25-34

---

## 🔍 验证检查点

在每个阶段完成后，运行以下命令验证：

```bash
# Phase 1 完成检查
uv run pytest lelamp/test/unit/test_auth_models.py -v
uv run pytest lelamp/test/integration/test_auth_service.py -v
uv run pytest lelamp/test/integration/test_auth_routes.py -v
uv run pytest lelamp/test/integration/test_auth_middleware.py -v
uv run pytest lelamp/test/integration/test_websocket_auth.py -v

# 整体覆盖率检查
uv run pytest --cov=lelamp --cov-report=term-missing

# 代码质量检查
uv run ruff check .
```

---

## 💡 重要提醒

1. **独立环境**: 这个 worktree 是完全独立的，不会影响主分支
2. **定期同步**: 完成重要里程碑后，考虑合并回 main 分支
3. **测试优先**: 每个功能都必须有对应的测试
4. **文档更新**: 重大变更需要更新 CLAUDE.md 和 README.md
5. **安全第一**: 涉及认证的代码要格外仔细

---

## 🎉 成功标准

- ✅ 所有测试通过 (目标: 80%+ 覆盖率)
- ✅ 代码质量检查通过 (ruff)
- ✅ 功能完整且安全
- ✅ 性能指标达标
- ✅ 文档完整更新

---

**准备就绪！请在新会话中开始执行优化计划。** 🚀
