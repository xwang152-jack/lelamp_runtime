# 文档更新摘要 (2026-03-25)

## 更新内容

### SETUP_GUIDE.md 更新

1. **版本信息更新**
   - 更新日期：2026-03-19 → 2026-03-25
   - 版本号：v0.1.1 → v0.1.2
   - 添加快速提示：预装系统设备可跳过部分步骤

2. **依赖安装部分更新**
   - 添加 `--extra api` 选项说明
   - 更新依赖分类说明：
     - `hardware`: LED 控制（GPIO 访问）
     - `vision`: 摄像头和视觉识别
     - `api`: FastAPI Web 服务和 REST API
   - 添加最小化安装选项

3. **服务架构部分大幅更新**
   - 添加运行模式对比表（4 种模式）
   - 新增 Captive Portal 模式说明
   - 更新服务启动方式：
     - **方式一**：使用启动脚本（推荐）
     - **方式二**：手动启动
   - 添加健康检查验证步骤

4. **新增服务管理部分**
   - **5.5 完整服务系统**：setup_all_services.sh
   - **5.6 Captive Portal**：开箱即用配置
   - **5.7 LiveKit Tmux 服务**：语音对话专用
   - 每个部分包含详细的安装、管理和使用说明

### USER_GUIDE_QUICK.md 新建

创建了全新的用户快速使用指南，包含：

1. **快速开始**
   - Web 界面控制步骤
   - 语音对话使用方法
   - LED 状态指示说明

2. **核心功能**
   - 语音对话（示例对话）
   - 动作表情（6 个预设动作）
   - 灯光效果（8 种纯色 + 6 种灯效）
   - 视觉识别（拍照识别、检查作业、飞书推送）
   - 系统管理（音量、搜索、OTA）

3. **服务管理**
   - 服务状态查看
   - 服务重启命令
   - 日志查看方法

4. **常见问题**
   - Q1: Web 界面无法访问
   - Q2: 语音不工作
   - Q3: LED 灯不亮
   - Q4: 获取台灯 IP 地址
   - Q5: 忘记 WiFi 密码

5. **首次开箱配置**
   - Captive Portal 配置流程
   - 重新进入设置模式方法

6. **高级功能**
   - 自定义动作录制
   - 边缘视觉（本地 AI）
   - Web 设置页面

7. **性能指标**
   - 延迟指标
   - 资源占用

### README.md 更新

1. **导航链接更新**
   - 更新为新文档结构
   - 突出显示用户快速指南

2. **功能描述更新**
   - 添加边缘视觉功能说明
   - 移除过时的 "⭐ NEW" 标记
   - 统一功能描述格式

3. **服务启动部分更新**
   - 移除过多的 "⭐" 标记
   - 保持清晰的三种方式说明
   - 更新示例 IP 地址

4. **文档链接部分重构**
   - 分为"新用户文档"、"开发文档"、"产品文档"、"技术文档"
   - 添加 USER_GUIDE_QUICK.md 链接
   - 更新文档结构

### .env.example 更新

添加边缘视觉配置：
```bash
LELAMP_EDGE_VISION_ENABLED=false
LELAMP_EDGE_VISION_MODEL_DIR=/home/pi/lelamp_runtime/models
LELAMP_EDGE_VISION_CONFIDENCE_THRESHOLD=0.5
LELAMP_EDGE_VISION_FPS=15
LELAMP_EDGE_VISION_BUFFER_SIZE=1
```

## 文档结构变化

### 之前
```
docs/
├── SETUP_GUIDE.md          # 详细配置教程
├── USER_GUIDE.md           # 完整使用指南（非常长）
├── CAPTIVE_PORTAL_GUIDE.md
└── ...
```

### 现在
```
docs/
├── SETUP_GUIDE.md          # 详细配置教程（已更新）
├── USER_GUIDE_QUICK.md     # 新用户快速指南（新建）
├── USER_GUIDE.md           # 完整使用指南（保留，供参考）
├── CAPTIVE_PORTAL_GUIDE.md
└── ...
```

## 主要改进

1. **降低新用户门槛**
   - 创建 USER_GUIDE_QUICK.md，快速上手
   - 突出显示常用功能
   - 简化配置步骤

2. **完善服务架构文档**
   - 详细说明 4 种运行模式
   - 添加 Captive Portal 配置流程
   - 完整的三服务架构说明

3. **统一文档风格**
   - 移除过多的版本标记
   - 统一格式和结构
   - 更新版本信息和日期

4. **增强可维护性**
   - 清晰的文档分层
   - 明确的受众定位
   - 完整的交叉引用

## 使用建议

### 对于新用户
1. 先阅读 [USER_GUIDE_QUICK.md](USER_GUIDE_QUICK.md)
2. 如需首次配置，参考 [SETUP_GUIDE.md](SETUP_GUIDE.md)
3. 遇到问题查看常见问题部分

### 对于开发者
1. 参考 [CLAUDE.md](../CLAUDE.md) 了解架构
2. 查看 [SETUP_GUIDE.md](SETUP_GUIDE.md) 了解配置
3. 阅读 API 文档进行集成开发

### 对于产品用户
1. 使用 Captive Portal 进行首次配置
2. 参考 [USER_GUIDE_QUICK.md](USER_GUIDE_QUICK.md) 学习使用
3. 遇到问题查看常见问题或联系支持

## 待办事项

- [ ] 更新英文版文档
- [ ] 添加视频教程链接
- [ ] 创建故障排查决策树
- [ ] 更新 API 文档链接
- [ ] 添加性能优化建议

## 版本历史

- **v0.1.2** (2026-03-25): 更新 SETUP_GUIDE，创建 USER_GUIDE_QUICK
- **v0.1.1** (2026-03-19): 初始版本，基础功能文档
