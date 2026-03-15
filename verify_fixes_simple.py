#!/usr/bin/env python3
"""
P1/P2 问题修复简化验证脚本
不依赖硬件模块，仅验证代码级别的修复
"""

import sys
import asyncio
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


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

        # 检查定义的内容
        if '"base_yaw": (-180, 180)' not in main_content:
            print("❌ 失败: SAFE_JOINT_RANGES 定义不正确")
            return False

        print("✅ SAFE_JOINT_RANGES 定义正确")

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


def test_p1_rate_limiter():
    """测试 P1-1: API 速率限制器"""
    print("🔍 测试 P1-1: API 速率限制器")

    try:
        # 检查文件是否存在
        rate_limiter_path = Path(__file__).parent / "lelamp" / "utils" / "rate_limiter.py"
        if not rate_limiter_path.exists():
            print("❌ 失败: rate_limiter.py 文件不存在")
            return False

        print("✅ rate_limiter.py 文件存在")

        # 检查文件内容
        rate_limiter_content = rate_limiter_path.read_text()

        # 检查关键类和函数
        if "class RateLimiter:" not in rate_limiter_content:
            print("❌ 失败: rate_limiter.py 未定义 RateLimiter 类")
            return False

        print("✅ RateLimiter 类已定义")

        if "async def acquire(" not in rate_limiter_content:
            print("❌ 失败: RateLimiter 缺少 acquire 方法")
            return False

        print("✅ RateLimiter 包含 acquire 方法")

        # 检查是否在 main.py 中使用
        main_py = Path(__file__).parent / "main.py"
        main_content = main_py.read_text()

        if "from lelamp.utils import get_rate_limiter" not in main_content:
            print("❌ 失败: main.py 未导入 get_rate_limiter")
            return False

        print("✅ main.py 导入了 get_rate_limiter")

        if "_search_rate_limiter = get_rate_limiter" not in main_content:
            print("❌ 失败: main.py 未创建速率限制器实例")
            return False

        print("✅ main.py 创建了速率限制器实例")

        if "await self._search_rate_limiter.acquire" not in main_content:
            print("❌ 失败: main.py 未使用速率限制器")
            return False

        print("✅ main.py 使用了速率限制器")
        print("✅ 测试 P1-1 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_p1_csv_cache():
    """测试 P1-4: CSV 缓存"""
    print("🔍 测试 P1-4: CSV 缓存")

    try:
        # 检查文件内容
        motors_service_path = Path(__file__).parent / "lelamp" / "service" / "motors" / "motors_service.py"
        motors_service_content = motors_service_path.read_text()

        # 检查缓存相关属性
        if "_recording_cache" not in motors_service_content:
            print("❌ 失败: motors_service.py 未定义 _recording_cache")
            return False

        print("✅ motors_service.py 包含录制数据缓存")

        # 检查缓存管理方法
        if "def clear_cache(self):" not in motors_service_content:
            print("❌ 失败: motors_service.py 缺少 clear_cache 方法")
            return False

        print("✅ motors_service.py 包含 clear_cache 方法")

        if "def get_cache_stats(self)" not in motors_service_content:
            print("❌ 失败: motors_service.py 缺少 get_cache_stats 方法")
            return False

        print("✅ motors_service.py 包含 get_cache_stats 方法")

        # 检查 _load_recording 方法是否使用缓存
        if "recording_name in self._recording_cache" not in motors_service_content:
            print("❌ 失败: _load_recording 未使用缓存")
            return False

        print("✅ _load_recording 使用了缓存")
        print("✅ 测试 P1-4 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
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

        baidu_auth_content = baidu_auth_path.read_text()

        # 检查关键类和方法
        if "class BaiduAuth:" not in baidu_auth_content:
            print("❌ 失败: baidu_auth.py 未定义 BaiduAuth 类")
            return False

        print("✅ BaiduAuth 类已定义")

        if "async def get_access_token(" not in baidu_auth_content:
            print("❌ 失败: BaiduAuth 缺少 get_access_token 方法")
            return False

        print("✅ BaiduAuth 包含 get_access_token 方法")

        if "def get_stats(self) -> dict:" not in baidu_auth_content:
            print("❌ 失败: BaiduAuth 缺少 get_stats 方法")
            return False

        print("✅ BaiduAuth 包含统计功能")

        # 检查是否有缓存逻辑
        if "_access_token_expires_at" not in baidu_auth_content:
            print("❌ 失败: BaiduAuth 缺少 Token 缓存逻辑")
            return False

        print("✅ BaiduAuth 包含 Token 缓存逻辑")

        # 检查 baidu_speech.py 是否使用共享认证
        baidu_speech_path = Path(__file__).parent / "lelamp" / "integrations" / "baidu_speech.py"
        baidu_speech_content = baidu_speech_path.read_text()

        if "from .baidu_auth import BaiduAuth" not in baidu_speech_content:
            print("❌ 失败: baidu_speech.py 未导入 BaiduAuth")
            return False

        print("✅ baidu_speech.py 使用了 BaiduAuth")

        if "self._auth = BaiduAuth(" not in baidu_speech_content:
            print("❌ 失败: baidu_speech.py 未创建 BaiduAuth 实例")
            return False

        print("✅ baidu_speech.py 创建了 BaiduAuth 实例")
        print("✅ 测试 P1-5 通过\n")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
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
    print("🔧 P1/P2 问题修复验证（简化版）")
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
