# 语音延迟优化与 TTS 缓存 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 启用 preemptive generation 降低响应延迟，并为固定短语实现 TTS 缓存减少百度 API 调用。

**Architecture:** 在 AgentSession 中加 `preemptive_generation=True`（1 行改动）。在 LeLamp agent 中新增 `_tts_cache` 字典，对 greeting、手势确认、电机故障提示等固定文本做 synthesize-once 缓存，命中时用 `session.say(text, audio=frames)` 绕过 TTS。

**Tech Stack:** LiveKit Agents SDK (Python), livekit.rtc.AudioFrame, BaiduTTS

---

### Task 1: 启用 Preemptive Generation

**Files:**
- Modify: `main.py:176-200`

**Step 1: 在 AgentSession 构造中添加 preemptive_generation=True**

在 `main.py` 第 196 行 `turn_handling` 之前添加一行：

```python
    session = AgentSession(
        vad=_build_vad(),
        stt=BaiduShortSpeechSTT(
            api_key=config.baidu_api_key,
            secret_key=config.baidu_secret_key,
            cuid=config.baidu_cuid,
            state_cb=_on_state,
            transcript_cb=_on_transcript,
        ),
        llm=deepseek_llm,
        tts=BaiduTTS(
            api_key=config.baidu_api_key,
            secret_key=config.baidu_secret_key,
            cuid=config.baidu_cuid,
            per=config.baidu_tts_per,
            state_cb=_on_state,
        ),
        preemptive_generation=True,  # 提前发起 LLM+TTS，降低响应延迟
        # 使用新的 turn_handling API (LiveKit 1.5+)
        # adaptive: ML 模型区分真正的打断 vs 假阳性（咳嗽、背景音等）
        # dynamic: 自适应沉默阈值，根据对话节奏动态调整
        turn_handling={
            "interruption": {"mode": "adaptive"},
            "endpointing": {"mode": "dynamic"},
        },
    )
```

**Step 2: 验证改动正确**

Run: `uv run python -c "import ast; ast.parse(open('main.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat(agent): 启用 preemptive generation 降低语音响应延迟"
```

---

### Task 2: 在 LeLamp 中添加 TTS 缓存基础设施

**Files:**
- Modify: `lelamp/agent/lelamp_agent.py:419-423`

**Step 1: 添加 _tts_cache 字段和固定短语列表**

在 `__init__` 中 `_motor_fault_notified` 之后（第 420 行附近）添加：

```python
        self._motor_fault_notified: dict = {}  # motor_name -> HealthStatus（Task 4 用）
        self._tts_cache: dict[str, list] = {}  # text -> list[rtc.AudioFrame]
```

**Step 2: 添加 _preheat_tts_cache 方法**

在 `_speak_proactively` 方法之前（第 519 行附近）添加新方法：

```python
    def _get_fixed_phrases(self) -> list[str]:
        """返回所有需要预热的固定短语"""
        phrases = []
        # 问候语
        greeting = os.getenv("LELAMP_GREETING_TEXT", "你好！我是 LeLamp，你的智能台灯。")
        if greeting:
            phrases.append(greeting)
        # 手势确认（所有可能的 gesture_name）
        gesture_names = ["点赞", "踩", "耶", "挥手", "握拳", "指向", "OK", "张开手掌"]
        for name in gesture_names:
            phrases.append(f"你是在比{name}吗？")
        # 电机故障提示
        phrases.append("我今天有点不舒服，动作可能不太灵活")
        phrases.append("我的关节好像有点问题，先凑合着用吧")
        return phrases

    async def _preheat_tts_cache(self) -> None:
        """后台预热固定短语的 TTS 缓存"""
        if not hasattr(self, "session") or self.session is None:
            return
        tts = self.session.tts
        if tts is None:
            return

        phrases = self._get_fixed_phrases()
        for text in phrases:
            if text in self._tts_cache:
                continue
            try:
                stream = tts.synthesize(text)
                frames: list = []
                async for event in stream:
                    frames.append(event.frame)
                if frames:
                    self._tts_cache[text] = frames
                    logger.debug(f"TTS cache preheated: {text[:20]}...")
            except Exception as e:
                logger.warning(f"TTS cache preheat failed for '{text[:20]}...': {e}")
```

**Step 3: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('lelamp/agent/lelamp_agent.py').read()); print('OK')"`
Expected: OK

---

### Task 3: 修改 _speak_proactively 使用缓存

**Files:**
- Modify: `lelamp/agent/lelamp_agent.py:519-528`

**Step 1: 修改 _speak_proactively 查缓存**

将现有 `_speak_proactively` 方法替换为：

```python
    async def _speak_proactively(self, text: str) -> None:
        """从异步上下文主动发声（手势确认、故障提示等）"""
        try:
            if hasattr(self, "session") and self.session is not None:
                cached = self._tts_cache.get(text)
                if cached:

                    async def _audio_gen():
                        for frame in cached:
                            yield frame

                    await self.session.say(text, audio=_audio_gen(), allow_interruptions=True)
                else:
                    await self.session.say(text, allow_interruptions=True)
            else:
                logger.info(f"[speak_proactively] session not ready: {text}")
        except Exception as e:
            logger.warning(f"Proactive speech failed: {e}")
```

**Step 2: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('lelamp/agent/lelamp_agent.py').read()); print('OK')"`
Expected: OK

---

### Task 4: 修改 main.py 使用缓存的 greeting 并触发预热

**Files:**
- Modify: `main.py:217-225`

**Step 1: 在 session.start 后预热 TTS 缓存并使用缓存 greeting**

将 `main.py` 中 `session.start` 之后的 greeting 逻辑替换为：

```python
    try:
        await session.start(
            agent=agent,
            room=ctx.room,
            **start_kwargs,
        )
        # 预热固定短语 TTS 缓存（后台，不阻塞 greeting）
        agent._track_task(agent._preheat_tts_cache())
        if config.greeting_text:
            cached = agent._tts_cache.get(config.greeting_text)
            if cached:

                async def _greeting_audio():
                    for frame in cached:
                        yield frame

                await session.say(config.greeting_text, audio=_greeting_audio(), allow_interruptions=False)
            else:
                # greeting 期间禁用打断
                await session.say(config.greeting_text, allow_interruptions=False)
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        logger.info("Session ended")
```

注意：greeting 文本可能在预热完成前就需要播放（预热是后台任务），所以 greeting 播放时先查缓存，未命中则走正常 TTS 路径。后续重复播放同一短语时即可命中缓存。

**Step 2: 验证语法**

Run: `uv run python -c "import ast; ast.parse(open('main.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add lelamp/agent/lelamp_agent.py main.py
git commit -m "feat(agent): 为固定短语添加 TTS 音频缓存，减少百度 API 调用"
```

---

### Task 5: 更新 CLAUDE.md 文档

**Files:**
- Modify: `CLAUDE.md`

**Step 1: 在 LiveKit SDK 相关段落中记录新特性**

在 CLAUDE.md 的 `**LiveKit 1.5+ turn_handling API**` 段落之后，添加：

```markdown
**Preemptive Generation** — `AgentSession` 启用 `preemptive_generation=True`，在用户说话过程中提前发起 LLM+TTS 请求，降低端到端响应延迟。代价是用户继续说话时预生成结果被丢弃，增加少量 token 消耗。

**TTS 固定短语缓存** — `LeLamp` agent 内部维护 `_tts_cache: dict[str, list[rtc.AudioFrame]]`，对 greeting、手势确认、电机故障提示等固定文本做 synthesize-once 缓存。命中时通过 `session.say(text, audio=frames)` 绕过百度 TTS API。预热通过 `_preheat_tts_cache()` 后台异步执行。
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: 记录 preemptive generation 和 TTS 缓存特性"
```
