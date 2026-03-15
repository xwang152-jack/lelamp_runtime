#!/usr/bin/env python3
"""
P1/P2 问题修复验证脚本
验证 P1 和 P2 优先级问题的修复效果
"""

import sys
import asyncio
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def test_p1_rate_limiter():
    """测试 P1-1: API 速率限制器"""
    print("🔍 测试 P1-1: API 速率限制器")

    try:
        from lelamp.utils import get_rate_limiter

        # 创建速率限制器
        limiter = get_rate_limiter("test", rate=10.0, capacity=5)
        print("✅ 速率限制器创建成功")

        # 测试令牌获取
        async def run_test():
            # 快速获取 10 个令牌（超过容量）
            allowed_count = 0
            denied_count = 0

            for i in range(10):
                if await limiter.acquire(tokens=1, timeout=0.1):
                    allowed_count += 1
                else:
                    denied_count += 1

            print(f"✅ 允许: {allowed_count}, 拒绝: {denied_count}")
            assert allowed_count <= 5, f"应该只允许 5 个请求，实际: {allowed_count}"
            assert denied_count >= 5, f"应该拒绝至少 5 个请求，实际: {denied_count}"

            # 测试统计信息
            stats = limiter.get_stats()
            print(f"✅ 统计: {stats}")

        asyncio.run(run_test())
        print("✅ 测试 P1-1 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_p1_async_subprocess():
    """测试 P1-2: 异步 subprocess 调用"""
    print("🔍 测试 P1-2: 异步 subprocess 调用")

    try:
        # 检查 main.py 中的异步调用
        main_py = Path(__file__).parent / "main.py"
        main_content = main_py.read_text()

        # 检查是否使用 asyncio.create_subprocess_exec
        if "asyncio.create_subprocess_exec" not in main_content:
            print("❌ 失败: main.py 未使用 asyncio.create_subprocess_exec")
            return False

        print("✅ main.py 使用了 asyncio.create_subprocess_exec")

        # 检查 _set_system_volume 是否是异步的
        if "async def _set_system_volume" not in main_content:
            print("❌ 失败: _set_system_volume 不是异步函数")
            return False

        print("✅ _set_system_volume 是异步函数")
        print("✅ 测试 P1-2 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_p1_input_validation():
    """测试 P1-3: 输入验证"""
    print("🔍 测试 P1-3: 输入验证（关节角度范围）")

    try:
        # 检查 SAFE_JOINT_RANGES 是否定义
        main_py = Path(__file__).parent / "main.py"
        main_content = main_py.read_text()

        if "SAFE_JOINT_RANGES" not in main_content:
            print("❌ 失败: main.py 未定义 SAFE_JOINT_RANGES")
            return False

        print("✅ SAFE_JOINT_RANGES 已定义")

        # 检查 move_joint 是否有角度验证
        if "超出安全范围" not in main_content and "angle_float < min_angle" not in main_content:
            print("❌ 失败: move_joint 缺少角度范围验证")
            return False

        print("✅ move_joint 包含角度范围验证")
        print("✅ 测试 P1-3 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_p1_csv_cache():
    """测试 P1-4: CSV 缓存"""
    print("🔍 测试 P1-4: CSV 缓存")

    try:
        from lelamp.service.motors.motors_service import MotorsService
        from unittest.mock import Mock, patch

        # Mock 机器人连接
        with patch('lelamp.service.motors.motors_service.LeLampFollower'):
            service = MotorsService(port="/dev/test", lamp_id="test")

        # 检查缓存相关属性
        if not hasattr(service, '_recording_cache'):
            print("❌ 失败: MotorsService 缺少 _recording_cache 属性")
            return False

        print("✅ MotorsService 包含录制数据缓存")

        # 检查缓存管理方法
        if not hasattr(service, 'clear_cache'):
            print("❌ 失败: MotorsService 缺少 clear_cache 方法")
            return False

        print("✅ MotorsService 包含 clear_cache 方法")

        if not hasattr(service, 'get_cache_stats'):
            print("❌ 失败: MotorsService 缺少 get_cache_stats 方法")
            return False

        print("✅ MotorsService 包含 get_cache_stats 方法")
        print("✅ 测试 P1-4 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_p1_shared_auth():
    """测试 P1-5: 共享认证类"""
    print("🔍 测试 P1-5: 共享认证类")

    try:
        # 检查共享认证类是否存在
        baidu_auth_path = Path(__file__).parent / "lelamp" / "integrations" / "baidu_auth.py"
        if not baidu_auth_path.exists():
            print("❌ 失败: baidu_auth.py 文件不存在")
            return False

        print("✅ baidu_auth.py 文件存在")

        # 尝试导入
        from lelamp.integrations.baidu_auth import BaiduAuth
        print("✅ BaiduAuth 类可以导入")

        # 检查是否有缓存和统计功能
        if not hasattr(BaiduAuth, 'get_access_token'):
            print("❌ 失败: BaiduAuth 缺少 get_access_token 方法")
            return False

        print("✅ BaiduAuth 包含 get_access_token 方法")

        if not hasattr(BaiduAuth, 'get_stats'):
            print("❌ 失败: BaiduAuth 缺少 get_stats 方法")
            return False

        print("✅ BaiduAuth 包含统计功能")

        # 检查 STT 是否使用共享认证
        baidu_speech_path = Path(__file__).parent / "lelamp" / "integrations" / "baidu_speech.py"
        baidu_speech_content = baidu_speech_path.read_text()

        if "from .baidu_auth import BaiduAuth" not in baidu_speech_content:
            print("❌ 失败: baidu_speech.py 未导入 BaiduAuth")
            return False

        print("✅ baidu_speech.py 使用了 BaiduAuth")
        print("✅ 测试 P1-5 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_p2_cache():
    """测试 P2-1: LLM 响应缓存"""
    print("🔍 测试 P2-1: LLM 响应缓存")

    try:
        from lelamp.cache import VisionCache, SearchCache

        # 测试视觉缓存
        vision_cache = VisionCache()
        print("✅ VisionCache 创建成功")

        # 测试缓存操作
        async def run_vision_test():
            # 测试缓存未命中
            result = await vision_cache.get("fake_image", "test question")
            assert result is None, "初始缓存应该为空"
            print("✅ 缓存未命中测试通过")

            # 测试设置缓存
            await vision_cache.set("fake_image", "test question", "test response", ttl_seconds=60)
            print("✅ 缓存设置成功")

            # 测试缓存命中
            result = await vision_cache.get("fake_image", "test question")
            assert result == "test response", f"缓存应该返回 'test response', 实际: {result}"
            print("✅ 缓存命中测试通过")

            # 测试统计
            stats = vision_cache.get_stats()
            print(f"✅ 缓存统计: {stats}")

            # 清理缓存
            await vision_cache.clear()

        asyncio.run(run_vision_test())

        # 测试搜索缓存
        search_cache = SearchCache()
        print("✅ SearchCache 创建成功")

        print("✅ 测试 P2-1 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("🔧 P1/P2 问题修复验证")
    print("=" * 60)
    print()

    results = []

    # P1 测试
    results.append(("P1-1: API 速率限制器", test_p1_rate_limiter()))
    results.append(("P1-2: 异步 subprocess", test_p1_async_subprocess()))
    results.append(("P1-3: 输入验证", test_p1_input_validation()))
    results.append(("P1-4: CSV 缓存", test_p1_csv_cache()))
    results.append(("P1-5: 共享认证", test_p1_shared_auth()))

    # P2 测试
    results.append(("P2-1: LLM 响应缓存", test_p2_cache()))

    # 打印总结
    print("=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    print()
    print(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有 P1/P2 问题修复验证通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
