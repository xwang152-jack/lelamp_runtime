"""
Agent 工具模块共享工具函数
"""

from typing import Tuple


def validate_rgb_color(r: int, g: int, b: int) -> Tuple[bool, str]:
    """
    验证 RGB 值是否在 0-255 范围内

    Args:
        r: 红色值
        g: 绿色值
        b: 蓝色值

    Returns:
        (is_valid, error_message) 元组
    """
    try:
        r_int = int(r)
        g_int = int(g)
        b_int = int(b)
    except (ValueError, TypeError):
        return False, "RGB 值必须是整数"

    if not all(0 <= val <= 255 for val in [r_int, g_int, b_int]):
        return False, "RGB 值必须在 0-255 范围内"

    return True, ""


def validate_multiple_rgb_colors(*colors: int) -> Tuple[bool, str]:
    """
    验证多个 RGB 值是否在 0-255 范围内

    Args:
        *colors: RGB 值列表（必须是 3 的倍数）

    Returns:
        (is_valid, error_message) 元组
    """
    if len(colors) % 3 != 0:
        return False, "RGB 值数量必须是 3 的倍数"

    try:
        colors_int = [int(c) for c in colors]
    except (ValueError, TypeError):
        return False, "RGB 值必须是整数"

    if not all(0 <= val <= 255 for val in colors_int):
        return False, "RGB 值必须在 0-255 范围内"

    return True, ""
