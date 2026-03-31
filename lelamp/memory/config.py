"""
记忆系统配置
"""
from dataclasses import dataclass

from lelamp.config import _get_env_bool, _get_env_int, _get_env_float


@dataclass(frozen=True)
class MemoryConfig:
    """记忆系统配置"""

    enabled: bool = True
    token_budget: int = 400
    consolidation_min_turns: int = 10
    consolidation_cooldown_s: float = 300.0
    max_content_length: int = 500


def load_memory_config() -> MemoryConfig:
    """从环境变量加载记忆配置"""
    return MemoryConfig(
        enabled=_get_env_bool("LELAMP_MEMORY_ENABLED", True),
        token_budget=_get_env_int("LELAMP_MEMORY_TOKEN_BUDGET", 400),
        consolidation_min_turns=_get_env_int(
            "LELAMP_MEMORY_CONSOLIDATION_MIN_TURNS", 10
        ),
        consolidation_cooldown_s=_get_env_float(
            "LELAMP_MEMORY_CONSOLIDATION_COOLDOWN_S", 300.0
        ),
        max_content_length=_get_env_int("LELAMP_MEMORY_MAX_CONTENT_LENGTH", 500),
    )
