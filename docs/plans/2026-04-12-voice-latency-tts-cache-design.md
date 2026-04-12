# 语音延迟优化与 TTS 缓存设计

日期：2026-04-12

## 背景

当前 LeLamp Agent 使用 STT-LLM-TTS pipeline，每次 TTS 都调用百度 API。固定短语（greeting、手势确认、故障提示）被反复合成，浪费 API 调用和响应时间。同时 Agent 未启用 preemptive generation，响应延迟未优化。

## 改进 1：Preemptive Generation

**改动文件**：`main.py`

在 `AgentSession` 构造中添加 `preemptive_generation=True`，使 Agent 在用户说话过程中（STT 产出中间 transcript 后）提前发起 LLM+TTS 请求，不等 end-of-turn 检测完成。

- 默认开启，无环境变量控制
- 代价：用户继续说话时预生成结果被丢弃，增加少量 token 消耗
- 改动量：1 行

## 改进 2：固定短语 TTS 缓存

**改动文件**：`lelamp/agent/lelamp_agent.py`, `main.py`

### 设计

在 `LeLamp` agent 内部维护 `dict[str, list[rtc.AudioFrame]]` 缓存，对固定短语跳过百度 TTS API 调用。

### 缓存范围

- `config.greeting_text`（启动问候语）
- `_speak_proactively` 中的手势确认文本
- 电机故障提示文本

### 实现要点

- 缓存 key：文本字符串本身
- 缓存内容：`tts.synthesize(text)` 收集的 PCM 音频帧
- 命中时使用 `session.say(text, audio=frames)` 绕过 TTS
- 预热时机：session 启动后、greeting 之前，后台异步预热
- 生命周期：session 级别，进程重启重新预热，无需持久化

### 影响范围

- `LeLamp.__init__`：新增 `_tts_cache: dict`
- 新增 `_preheat_tts_cache()`：异步预热方法
- `_speak_proactively()`：先查缓存，命中则 `say(text, audio=...)`
- `main.py`：greeting 使用缓存帧
