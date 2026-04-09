"""
记忆整合器 — LLM 驱动的记忆提取和对话摘要

参考 nanobot 的 MemoryConsolidator 设计：
1. 当对话轮数达到阈值时触发整合
2. 调用 DeepSeek API 从对话中提取关键信息
3. 去重后存入长期记忆
4. 生成对话摘要供后续搜索

使用 httpx 调用 DeepSeek API，复用 @retry_on_error + @with_fallback 模式。
"""
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

import httpx

from lelamp.memory.store import MemoryStore, _CHARS_PER_TOKEN
from lelamp.memory.config import load_memory_config

logger = logging.getLogger("lelamp.memory.consolidator")

# 整合 prompt
_CONSOLIDATION_SYSTEM = """你是一个记忆管理系统。你的任务是从对话记录中提取值得长期记住的信息。

规则：
1. 只提取用户明确表达的偏好、事实、关系和重要上下文
2. 不要提取临时性信息（如"今天天气不错"）
3. 不要提取已经很明显的信息
4. 如果新信息与旧信息矛盾（例如偏好改变），在 memories 中包含新信息，并在 obsolete_ids 中列出应该失效的旧记忆 id
5. 每条记忆控制在 100 字以内

请返回 JSON 格式：
{
  "memories": [
    {"content": "用户偏好冷色调灯光", "category": "preference", "importance": 7}
  ],
  "obsolete_ids": [123],
  "summary": "本次对话的简短摘要（2-3句话）",
  "topics": ["关键词1", "关键词2"]
}

category 可选值: preference(偏好), fact(事实), relationship(关系), context(上下文), general(通用)
importance 范围: 1-10，越高越重要
obsolete_ids: 应该失效的旧记忆 id 列表（整数），没有则返回空数组 []

如果没有值得记住的信息，返回空的 memories 数组。"""


@dataclass
class ConsolidationResult:
    """整合结果"""
    new_memories_count: int = 0
    summary_saved: bool = False
    topics: list[str] = field(default_factory=list)


class MemoryConsolidator:
    """LLM 驱动的记忆整合器"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        memory_store: MemoryStore,
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._store = memory_store
        self._timeout_s = timeout_s

        self._lock = threading.Lock()
        self._last_consolidation_ts: float = 0.0

        config = load_memory_config()
        self._min_turns = config.consolidation_min_turns
        self._cooldown_s = config.consolidation_cooldown_s
        self._context_token_limit = config.context_token_limit
        self._context_budget_ratio = config.context_budget_ratio

        self._llm_failure_count: int = 0        # 连续 LLM 调用失败次数
        self._max_llm_failures: int = 3         # 超过此数触发降级

    def should_consolidate(
        self,
        conversation_turns: list[dict],
        *,
        min_turns: int | None = None,
        cooldown_s: float | None = None,
    ) -> bool:
        """检查是否应该触发整合（原子操作，通过即设置时间戳）"""
        if min_turns is None:
            min_turns = self._min_turns
        if cooldown_s is None:
            cooldown_s = self._cooldown_s

        if len(conversation_turns) < min_turns:
            return False

        with self._lock:
            if time.time() - self._last_consolidation_ts < cooldown_s:
                return False
            # 原子设置：通过检查后立即标记，防止并发重复触发
            self._last_consolidation_ts = time.time()
            return True

    def should_consolidate_by_tokens(
        self,
        conversation_turns: list[dict],
    ) -> bool:
        """
        基于 token 估算判断是否应触发整合。

        当未整合轮次的估算 token 数超过 context 预算阈值时触发，
        与 should_consolidate（轮数）互补，两者满足其一即整合。
        中文约 0.67 chars/token。
        """
        if not conversation_turns:
            return False

        estimated = sum(len(t.get("content", "")) / _CHARS_PER_TOKEN for t in conversation_turns)
        threshold = self._context_token_limit * self._context_budget_ratio

        if estimated > threshold:
            with self._lock:
                if time.time() - self._last_consolidation_ts < self._cooldown_s:
                    return False  # 仍在冷却中
                self._last_consolidation_ts = time.time()
                return True
        return False

    async def consolidate(
        self,
        lamp_id: str,
        session_id: str,
        conversation_turns: list[dict],
    ) -> Optional[ConsolidationResult]:
        """
        执行记忆整合

        1. 构建包含已有记忆的 prompt
        2. 调用 DeepSeek API
        3. 解析 JSON 响应
        4. 去重并保存新记忆
        5. 保存对话摘要
        """
        # 调用方负责传入未整合的轮次；内部最多保留 40 条防止 prompt 过大
        recent_turns = conversation_turns[-40:]

        # 获取已有记忆用于去重
        existing = self._store.get_all_active_memories(lamp_id)

        # 构建 user prompt
        existing_text = ""
        if existing:
            lines = ["已有记忆（避免重复，如有矛盾请列入 obsolete_ids）:"]
            for m in existing:
                lines.append(f"- [id={m.id}] [{m.category}] {m.content}")
            existing_text = "\n".join(lines) + "\n\n"

        turns_text = "\n".join(
            f"{t['role']}: {t['content']}" for t in recent_turns
        )

        user_prompt = f"{existing_text}最近对话记录:\n{turns_text}"

        # 调用 LLM
        llm_response = await self._call_llm(_CONSOLIDATION_SYSTEM, user_prompt)
        if not llm_response:
            with self._lock:
                self._llm_failure_count += 1
                should_fallback = self._llm_failure_count >= self._max_llm_failures
                if should_fallback:
                    self._llm_failure_count = 0
            if should_fallback:
                logger.warning(
                    f"LLM consolidation failed {self._max_llm_failures} times, "
                    "falling back to raw archive"
                )
                return self._raw_archive(lamp_id, session_id, recent_turns)
            return None

        # LLM 成功：重置失败计数
        with self._lock:
            self._llm_failure_count = 0

        # 解析 JSON 响应
        try:
            data = json.loads(llm_response)
        except json.JSONDecodeError:
            # 尝试从 markdown code block 中提取 JSON
            llm_response = llm_response.strip()
            if llm_response.startswith("```"):
                lines = llm_response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        if in_block:
                            break
                        in_block = True
                        continue
                    if in_block:
                        json_lines.append(line)
                try:
                    data = json.loads("\n".join(json_lines))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse consolidation response as JSON")
                    return None
            else:
                logger.warning("Failed to parse consolidation response as JSON")
                return None

        # 处理矛盾记忆失效
        obsolete_ids = data.get("obsolete_ids", [])
        if obsolete_ids:
            for oid in obsolete_ids:
                try:
                    self._store.deactivate_memory(int(oid))
                except (ValueError, TypeError, Exception) as e:
                    logger.warning(f"Failed to deactivate obsolete memory id={oid}: {e}")
            logger.info(f"Deactivated {len(obsolete_ids)} obsolete memories due to contradiction")

        # 提取记忆
        new_memories = data.get("memories", [])
        deduped = self._deduplicate_memories(new_memories, existing)

        result = ConsolidationResult(
            new_memories_count=len(deduped),
            topics=data.get("topics", []),
        )

        # 保存新记忆
        for mem in deduped:
            try:
                self._store.add_memory(
                    lamp_id=lamp_id,
                    content=mem["content"],
                    category=mem.get("category", "general"),
                    importance=mem.get("importance", 5),
                    source="auto",
                )
            except Exception as e:
                logger.error(f"Failed to save consolidated memory: {e}")

        # 保存对话摘要
        summary = data.get("summary", "")
        if summary:
            try:
                now = datetime.now(UTC)
                self._store.save_summary(
                    lamp_id=lamp_id,
                    session_id=session_id,
                    summary=summary,
                    key_topics=data.get("topics", []),
                    message_count=len(recent_turns),
                    started_at=now,
                    ended_at=now,
                )
                result.summary_saved = True
            except Exception as e:
                logger.error(f"Failed to save conversation summary: {e}")

        logger.info(
            f"Consolidation complete: {result.new_memories_count} new memories, "
            f"summary_saved={result.summary_saved}"
        )
        return result

    async def _call_llm(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[str]:
        """调用 DeepSeek API（带重试和降级）"""
        if not self._api_key:
            logger.warning("No API key for memory consolidation")
            return None

        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            choice0 = (data.get("choices") or [])[0]
            msg = (choice0 or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()

            return None
        except Exception as e:
            logger.warning(f"LLM call for consolidation failed: {e}")
            return None

    def _raw_archive(
        self,
        lamp_id: str,
        session_id: str,
        turns: list[dict],
    ) -> ConsolidationResult:
        """
        降级存储：LLM 多次失败时直接把对话拼接为摘要存入 ConversationSummary。
        不提取长期记忆，仅保留对话历史以防数据丢失。
        """
        user_lines = [
            t["content"][:80]  # 每条用户发言截取 80 字符（约 80 汉字）
            for t in turns
            if t.get("role") == "user" and t.get("content")
        ]
        fallback_summary = "【自动降级摘要】" + " | ".join(user_lines[:5])  # 最多 5 条

        try:
            now = datetime.now(UTC)
            self._store.save_summary(
                lamp_id=lamp_id,
                session_id=session_id,
                summary=fallback_summary[:500],
                key_topics=[],
                message_count=len(turns),
                started_at=now,
                ended_at=now,
            )
            logger.info(f"Raw archive saved for session {session_id}")
            return ConsolidationResult(new_memories_count=0, summary_saved=True)
        except Exception as e:
            logger.error(f"Raw archive failed: {e}")
            return ConsolidationResult(new_memories_count=0, summary_saved=False)

    def _deduplicate_memories(
        self,
        new_memories: list[dict],
        existing: list,
    ) -> list[dict]:
        """
        去重：新记忆与已有记忆内容重叠 >60% 时跳过。

        使用 bigram 重叠比例判断，对中文更准确。
        """
        deduped: list[dict] = []
        for new_mem in new_memories:
            content = new_mem.get("content", "").strip()
            if not content:
                continue

            is_dup = False
            for existing_mem in existing:
                overlap = self._content_overlap(content, existing_mem.content)
                if overlap > 0.6:
                    is_dup = True
                    # 如果新记忆更重要，更新已有记忆
                    new_importance = new_mem.get("importance", 5)
                    if new_importance > existing_mem.importance:
                        self._store.update_memory(
                            existing_mem.id,
                            content=content,
                            importance=new_importance,
                        )
                    break

            if not is_dup:
                deduped.append(new_mem)

        return deduped

    @staticmethod
    def _content_overlap(a: str, b: str) -> float:
        if not a or not b:
            return 0.0

        def bigrams(s: str) -> set[str]:
            return {s[i:i+2] for i in range(len(s) - 1)}

        bg_a = bigrams(a)
        bg_b = bigrams(b)
        if not bg_a or not bg_b:
            return 0.0

        intersection = bg_a & bg_b
        union = bg_a | bg_b
        return len(intersection) / len(union)
