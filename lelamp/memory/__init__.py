"""
记忆系统包

双层记忆架构（参考 nanobot）：
- 长期记忆 (Memory): 注入 system prompt，持久化用户偏好/事实
- 对话摘要 (ConversationSummary): 按需搜索的历史对话记录
"""
from lelamp.memory.config import MemoryConfig, load_memory_config
from lelamp.memory.models import Memory, ConversationSummary
from lelamp.memory.store import MemoryStore

# 导入 consolidator 需要的可选依赖
try:
    from lelamp.memory.consolidator import MemoryConsolidator
except ImportError:
    MemoryConsolidator = None  # type: ignore

__all__ = [
    "Memory",
    "ConversationSummary",
    "MemoryStore",
    "MemoryConsolidator",
    "MemoryConfig",
    "load_memory_config",
]
