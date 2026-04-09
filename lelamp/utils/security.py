import hashlib
import hmac
import os
import uuid
import platform
import logging

logger = logging.getLogger("lelamp.security")

def get_device_id() -> str:
    """
    获取设备唯一标识符 (Device ID).
    在 Linux (树莓派/Jetson) 上尝试读取 CPU Serial，
    在 macOS/Windows 上回退到 MAC 地址。
    """
    try:
        # 尝试读取树莓派/Jetson CPU 序列号
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("Serial"):
                        return line.split(":")[1].strip()
        
        # 回退方案：使用 MAC 地址
        mac = uuid.getnode()
        return str(mac)
    except Exception as e:
        logger.warning(f"无法获取设备 ID: {e}")
        return "unknown_device"

def generate_license_key(device_id: str, secret: str | None = None) -> str:
    """
    基于设备 ID 生成授权码。
    使用 HMAC-SHA256 进行签名，增强安全性。
    实际商用应使用非对称加密 (RSA/ECC) 签名。

    Args:
        device_id: 设备唯一标识符
        secret: 签名密钥。如未提供，从环境变量 LELAMP_LICENSE_SECRET 读取

    Returns:
        16 字符的授权码

    Raises:
        RuntimeError: 如果未设置 LELAMP_LICENSE_SECRET 环境变量
    """
    if secret is None:
        secret = os.getenv("LELAMP_LICENSE_SECRET")
        if not secret:
            raise RuntimeError(
                "LELAMP_LICENSE_SECRET 环境变量未设置。"
                "请在 .env 文件中配置此密钥（生产环境必须使用随机强密钥）"
            )

    # 使用 HMAC 而不是简单的 SHA256
    signature = hmac.new(
        secret.encode(),
        device_id.encode(),
        hashlib.sha256
    ).hexdigest()[:16]
    return signature

def verify_license() -> bool:
    """
    校验当前设备是否已授权。
    检查环境变量 LELAMP_LICENSE_KEY 是否匹配设备 ID。

    开发模式：设置 LELAMP_DEV_MODE=1 跳过授权检查
    """
    # 开发模式跳过检查
    if os.getenv("LELAMP_DEV_MODE", "").lower() in ("1", "true", "yes"):
        logger.warning("开发模式：跳过许可证验证")
        return True

    device_id = get_device_id()
    provided_key = os.getenv("LELAMP_LICENSE_KEY")

    if not provided_key:
        logger.error(f"未找到授权码 (LELAMP_LICENSE_KEY)。设备 ID: {device_id}")
        # 生产环境不应输出期望的 key
        if os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG":
            try:
                expected_key = generate_license_key(device_id)
                logger.debug(f"开发模式建议授权码: {expected_key}")
            except RuntimeError:
                logger.debug("无法生成授权码：LELAMP_LICENSE_SECRET 未设置")
        return False

    try:
        expected_key = generate_license_key(device_id)
    except RuntimeError as e:
        logger.error(f"无法验证授权码: {e}")
        return False

    if not hmac.compare_digest(provided_key, expected_key):
        logger.error(f"授权码无效！设备 ID: {device_id}")
        # 不输出提供的授权码（避免日志泄露）
        return False

    logger.info(f"设备授权校验通过 (Device ID: {device_id})")
    return True
