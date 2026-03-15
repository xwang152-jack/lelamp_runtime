import hashlib
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

def generate_license_key(device_id: str, secret: str = "lelamp-secret-salt") -> str:
    """
    基于设备 ID 生成简单的授权码 (用于演示).
    实际商用应使用非对称加密 (RSA/ECC) 签名。
    """
    payload = f"{device_id}:{secret}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def verify_license() -> bool:
    """
    校验当前设备是否已授权。
    检查环境变量 LELAMP_LICENSE_KEY 是否匹配设备 ID。
    """
    device_id = get_device_id()
    expected_key = generate_license_key(device_id)
    provided_key = os.getenv("LELAMP_LICENSE_KEY")

    if not provided_key:
        logger.warning(f"未找到授权码 (LELAMP_LICENSE_KEY)。设备 ID: {device_id}")
        logger.warning(f"开发模式建议授权码: {expected_key}")
        return False

    if provided_key != expected_key:
        logger.error(f"授权码无效！设备 ID: {device_id}, 提供的授权码: {provided_key}")
        return False
    
    logger.info(f"设备授权校验通过 (Device ID: {device_id})")
    return True
