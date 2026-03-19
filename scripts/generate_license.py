#!/usr/bin/env python3
"""
LeLamp 设备授权码生成工具

用于商业化部署时生成设备许可证密钥。

使用场景：
1. 本地商业化版本（Console 模式）- 只需许可证验证，不需要 LiveKit
2. 远程商业化版本（Room 模式）- 需要许可证验证 + LiveKit

生成步骤：
1. 生成强随机密钥（LELAMP_LICENSE_SECRET）- 只需生成一次，务必保密！
2. 获取设备 ID
3. 生成该设备的授权码（LELAMP_LICENSE_KEY）
4. 将生成的授权码添加到 .env 文件

安全警告：
- LELAMP_LICENSE_SECRET 必须保密，不能泄露给最终用户
- 授权码可以安全地分发给设备用户
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lelamp.utils.security import get_device_id, generate_license_key
from dotenv import load_dotenv

load_dotenv()


def print_header(text: str) -> None:
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str) -> None:
    """打印小节"""
    print(f"\n--- {text} ---")


def generate_secret() -> str:
    """生成强随机密钥（64 字符十六进制）"""
    import secrets
    return secrets.token_hex(32)


def main():
    parser = argparse.ArgumentParser(
        description="LeLamp 设备授权码生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 查看当前设备 ID
  python scripts/generate_license.py --device-id

  # 生成新的密钥（只需执行一次）
  python scripts/generate_license.py --generate-secret

  # 为指定设备生成授权码
  python scripts/generate_license.py --device-id 00e04c123456

  # 交互式向导
  python scripts/generate_license.py --wizard

  # 验证当前 .env 配置
  python scripts/generate_license.py --verify
        """
    )

    parser.add_argument(
        "--device-id",
        action="store_true",
        help="显示当前设备的 ID"
    )

    parser.add_argument(
        "--generate-secret",
        action="store_true",
        help="生成新的强随机密钥（LELAMP_LICENSE_SECRET）"
    )

    parser.add_argument(
        "--license-for",
        metavar="DEVICE_ID",
        help="为指定设备 ID 生成授权码"
    )

    parser.add_argument(
        "--secret",
        metavar="SECRET",
        help="指定签名密钥（默认从环境变量读取）"
    )

    parser.add_argument(
        "--wizard",
        action="store_true",
        help="启动交互式配置向导"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证当前 .env 配置是否正确"
    )

    parser.add_argument(
        "--dev-mode",
        choices=["0", "1"],
        help="设置开发模式（0=生产模式需验证，1=开发模式跳过验证）"
    )

    args = parser.parse_args()

    # 如果没有指定任何操作，显示帮助
    if not any([
        args.device_id,
        args.generate_secret,
        args.license_for,
        args.wizard,
        args.verify,
        args.dev_mode
    ]):
        parser.print_help()
        return

    # 1. 显示设备 ID
    if args.device_id:
        print_header("设备 ID 信息")
        device_id = get_device_id()
        print(f"\n当前设备 ID: {device_id}")
        print("\n说明：")
        print("  - Linux (树莓派): 读取 CPU 序列号")
        print("  - macOS/Windows: 使用 MAC 地址")
        return

    # 2. 生成新密钥
    if args.generate_secret:
        print_header("生成新的签名密钥")
        print("\n⚠️  警告：此密钥必须保密！不要泄露给最终用户。")
        new_secret = generate_secret()
        print(f"\n生成的密钥（LELAMP_LICENSE_SECRET）：")
        print(f"\n{new_secret}")
        print("\n请将此密钥添加到 .env 文件中：")
        print("LELAMP_LICENSE_SECRET=" + new_secret)
        return

    # 3. 为指定设备生成授权码
    if args.license_for:
        device_id = args.license_for
        secret = args.secret or os.getenv("LELAMP_LICENSE_SECRET")

        print_header("生成设备授权码")
        print(f"\n设备 ID: {device_id}")

        if not secret:
            print("\n❌ 错误：未找到签名密钥")
            print("\n请通过以下方式之一提供密钥：")
            print("  1. 设置环境变量: export LELAMP_LICENSE_SECRET=your_secret")
            print("  2. 在 .env 文件中添加: LELAMP_LICENSE_SECRET=your_secret")
            print("  3. 使用命令行参数: --secret your_secret")
            sys.exit(1)

        try:
            license_key = generate_license_key(device_id, secret)
            print(f"\n✅ 授权码生成成功！")
            print(f"\n授权码（LELAMP_LICENSE_KEY）：")
            print(f"\n{license_key}")
            print("\n请将此授权码添加到目标设备的 .env 文件中：")
            print(f"LELAMP_LICENSE_KEY={license_key}")
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")
            sys.exit(1)
        return

    # 4. 交互式向导
    if args.wizard:
        print_header("LeLamp 商业化配置向导")

        print("\n此向导将帮助您完成商业化配置。")
        print("\n请选择配置类型：")
        print("  1. 本地版本（Console 模式）- 台灯本机操作，不需要远程访问")
        print("  2. 远程版本（Room 模式）- 支持手机 App 远程控制")

        mode = input("\n请选择 (1/2) [默认: 1]: ").strip() or "1"

        if mode == "1":
            print("\n📍 配置类型：本地商业化版本")
            print("\n必需的 API 配置：")
            print("  - DEEPSEEK_API_KEY")
            print("  - BAIDU_SPEECH_API_KEY")
            print("  - BAIDU_SPEECH_SECRET_KEY")
        else:
            print("\n📍 配置类型：远程商业化版本")
            print("\n必需的 API 配置：")
            print("  - LIVEKIT_URL")
            print("  - LIVEKIT_API_KEY")
            print("  - LIVEKIT_API_SECRET")
            print("  - DEEPSEEK_API_KEY")
            print("  - BAIDU_SPEECH_API_KEY")
            print("  - BAIDU_SPEECH_SECRET_KEY")

        # 步骤 1：检查密钥
        print_section("步骤 1：检查签名密钥")
        existing_secret = os.getenv("LELAMP_LICENSE_SECRET")

        if existing_secret:
            print(f"✅ 已找到签名密钥: {existing_secret[:8]}...{existing_secret[-8:]}")
            use_existing = input("\n是否使用现有密钥？(y/n) [默认: y]: ").strip().lower() or "y"
            if use_existing != "y":
                existing_secret = None

        if not existing_secret:
            print("\n生成新的签名密钥...")
            new_secret = generate_secret()
            print(f"\n生成的密钥: {new_secret}")
            confirm = input("\n是否保存此密钥到 .env 文件？(y/n) [默认: y]: ").strip().lower() or "y"

            if confirm == "y":
                env_path = project_root / ".env"
                with open(env_path, "a") as f:
                    f.write(f"\nLELAMP_LICENSE_SECRET={new_secret}\n")
                print(f"✅ 已添加到 {env_path}")
                existing_secret = new_secret
            else:
                print("\n❌ 未保存密钥，请手动添加到 .env 文件")
                return

        # 步骤 2：获取设备 ID
        print_section("步骤 2：获取设备信息")
        current_device_id = get_device_id()
        print(f"\n当前设备 ID: {current_device_id}")

        # 步骤 3：生成授权码
        print_section("步骤 3：生成授权码")

        ask_for_other = input("\n是否为其他设备生成授权码？(y/n) [默认: n]: ").strip().lower() or "n"

        if ask_for_other == "y":
            target_device_id = input("请输入目标设备 ID: ").strip()
        else:
            target_device_id = current_device_id

        try:
            license_key = generate_license_key(target_device_id, existing_secret)
            print(f"\n✅ 授权码生成成功！")
            print(f"\n设备 ID: {target_device_id}")
            print(f"授权码: {license_key}")
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")
            sys.exit(1)

        # 步骤 4：保存授权码
        print_section("步骤 4：保存配置")

        save_license = input("\n是否将授权码保存到 .env 文件？(y/n) [默认: y]: ").strip().lower() or "y"

        if save_license == "y":
            env_path = project_root / ".env"

            # 检查是否已存在 LELAMP_LICENSE_KEY
            env_content = ""
            if env_path.exists():
                with open(env_path, "r") as f:
                    env_content = f.read()

            # 更新或添加
            lines = env_content.split("\n")
            updated = False
            for i, line in enumerate(lines):
                if line.startswith("LELAMP_LICENSE_KEY="):
                    lines[i] = f"LELAMP_LICENSE_KEY={license_key}"
                    updated = True
                    break

            if not updated:
                lines.append(f"LELAMP_LICENSE_KEY={license_key}")

            # 关闭开发模式
            for i, line in enumerate(lines):
                if line.startswith("LELAMP_DEV_MODE="):
                    lines[i] = "LELAMP_DEV_MODE=0"
                    updated = True
                    break

            if not any(line.startswith("LELAMP_DEV_MODE=") for line in lines):
                lines.append("LELAMP_DEV_MODE=0")

            with open(env_path, "w") as f:
                f.write("\n".join(lines))

            print(f"✅ 配置已保存到 {env_path}")
        else:
            print("\n请手动添加以下内容到 .env 文件：")
            print(f"LELAMP_LICENSE_KEY={license_key}")
            print("LELAMP_DEV_MODE=0")

        # 完成
        print_header("配置完成")
        print("\n✅ 商业化配置已完成！")
        print("\n启动方式：")
        if mode == "1":
            print("  sudo uv run main.py console")
        else:
            print("  sudo uv run main.py dev  # 或使用 room 连接")
        print()

        return

    # 5. 验证配置
    if args.verify:
        print_header("验证商业化配置")

        # 检查密钥
        secret = os.getenv("LELAMP_LICENSE_SECRET")
        if secret:
            print(f"✅ LELAMP_LICENSE_SECRET: {secret[:8]}...{secret[-8:]}")
        else:
            print("❌ LELAMP_LICENSE_SECRET: 未设置")

        # 检查授权码
        license_key = os.getenv("LELAMP_LICENSE_KEY")
        if license_key:
            print(f"✅ LELAMP_LICENSE_KEY: {license_key}")
        else:
            print("❌ LELAMP_LICENSE_KEY: 未设置")

        # 检查开发模式
        dev_mode = os.getenv("LELAMP_DEV_MODE", "0")
        if dev_mode in ("1", "true", "yes"):
            print(f"⚠️  LELAMP_DEV_MODE: {dev_mode} (开发模式，跳过授权检查)")
        else:
            print(f"✅ LELAMP_DEV_MODE: {dev_mode} (生产模式)")

        # 验证授权
        print_section("授权验证")
        try:
            from lelamp.utils.security import verify_license
            if verify_license():
                print("✅ 授权验证通过")
            else:
                print("❌ 授权验证失败")
                sys.exit(1)
        except Exception as e:
            print(f"❌ 验证过程出错: {e}")
            sys.exit(1)

        return

    # 6. 设置开发模式
    if args.dev_mode:
        env_path = project_root / ".env"

        if not env_path.exists():
            print(f"\n❌ 错误：.env 文件不存在: {env_path}")
            print("请先创建 .env 文件: cp .env.example .env")
            sys.exit(1)

        # 读取并更新
        with open(env_path, "r") as f:
            content = f.read()

        lines = content.split("\n")
        updated = False

        for i, line in enumerate(lines):
            if line.startswith("LELAMP_DEV_MODE="):
                lines[i] = f"LELAMP_DEV_MODE={args.dev_mode}"
                updated = True
                break

        if not updated:
            lines.append(f"LELAMP_DEV_MODE={args.dev_mode}")

        with open(env_path, "w") as f:
            f.write("\n".join(lines))

        status = "开发模式（跳过授权检查）" if args.dev_mode == "1" else "生产模式（需要授权验证）"
        print(f"\n✅ 已设置为: {status}")
        return


if __name__ == "__main__":
    main()
