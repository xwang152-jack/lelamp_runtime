# 🚀 新会话启动指令

**复制以下内容到新的 Claude Code 会话中：**

---

## 会话设置指令

```
# 📍 工作目录设置
cd /Users/jackwang/lelamp_runtime/.claude/worktrees/optimization-implementation

# 🎯 执行目标
我将执行 LeLamp Runtime 的全面优化计划。

## 📋 实施计划位置
详细实施计划：docs/plans/2025-03-19-comprehensive-optimization-plan.md

## 🚀 执行要求
请使用 superpowers:executing-plans 技能来执行这个优化计划。

## 📊 计划概览
这是一个三阶段的渐进式优化计划：

### Phase 1: 安全基础设施 (1-2 周) - 当前阶段
- Task 1: 创建用户认证基础模型
- Task 2: 实现 JWT 认证服务
- Task 3: 创建认证 API 路由
- Task 4: 创建认证中间件
- Task 5: WebSocket 认证集成

### Phase 2: 功能完善 (2-3 周)
- 用户管理界面
- 视觉功能前端
- 设备管理功能

### Phase 3: 性能优化 (1-2 周)
- 数据库优化
- API 性能提升
- 前端性能优化

## ⚡ 执行原则
1. **TDD 驱动**: 先写测试，再实现功能
2. **小步快跑**: 每个 Step 都是 2-5 分钟的小任务
3. **频繁提交**: 每个 Step 完成后都 git commit
4. **持续验证**: 运行测试确保质量

## 🎯 成功标准
- 所有测试通过 (目标 80%+ 覆盖率)
- 代码质量检查通过 (ruff)
- 功能完整且安全
- 性能指标达标

## 📝 开始执行
请从 Phase 1 的 Task 1 开始，按照计划中的详细步骤逐个执行。
每个任务都有完整的代码示例、测试步骤和提交指令。

让我们开始吧！
```

---

## 🛠️ 环境准备指令

在新会话中，您可能还需要运行：

```bash
# 确认工作目录
pwd
# 应该显示: /Users/jackwang/lelamp_runtime/.claude/worktrees/optimization-implementation

# 确认分支
git branch
# 应该显示: * optimization-implementation

# 检查环境
./start-execution.sh
```

---

## 📋 快速参考

**如果新会话中遇到问题，请告诉它：**

1. **查看执行指南**: `cat EXECUTION_SESSION.md`
2. **查看详细计划**: `cat docs/plans/2025-03-19-comprehensive-optimization-plan.md`
3. **遵循 TDD 原则**: 先写测试，再实现功能
4. **小步快跑**: 每个 Step 都要完成并提交

**验证执行进度:**
```bash
# 查看提交历史
git log --oneline --since="1 hour ago"

# 运行测试
uv run pytest lelamp/test/unit/test_auth_models.py -v

# 检查覆盖率
uv run pytest --cov=lelamp --cov-report=term-missing
```

---

## 🎉 准备启动

**现在您可以：**

1. **启动新的 Claude Code 会话** (新终端或新窗口)
2. **切换到执行目录**: `cd /Users/jackwang/lelamp_runtime/.claude/worktrees/optimization-implementation`
3. **粘贴上面的会话设置指令**
4. **开始执行优化计划！**

**祝执行顺利！** 🚀
