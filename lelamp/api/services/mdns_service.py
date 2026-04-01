"""
mDNS/Bonjour 设备自动发现服务

通过 zeroconf 注册 _http._tcp 服务，使用户可以通过
http://lelamp.local:8000 访问设备，无需记忆 IP 地址。
"""
import logging

logger = logging.getLogger("lelamp.api.mdns")


class MDNSService:
    """mDNS 服务注册/注销"""

    def __init__(self, device_id: str = "lelamp", port: int = 8000):
        self.device_id = device_id
        self.port = port
        self._zeroconf = None
        self._service_info = None

    def register(self) -> None:
        """注册 mDNS 服务，失败时静默降级。"""
        try:
            from zeroconf import Zeroconf, ServiceInfo

            service_name = f"LeLamp {self.device_id}"
            service_type = "_http._tcp.local."

            self._service_info = ServiceInfo(
                service_type,
                f"{service_name}.{service_type}",
                addresses=None,  # 自动绑定所有接口
                port=self.port,
                properties={
                    "device_id": self.device_id,
                    "model": "LeLamp",
                    "version": "0.1.0",
                },
            )

            self._zeroconf = Zeroconf()
            self._zeroconf.register_service(self._service_info)
            logger.info(
                f"mDNS registered: {service_name} on port {self.port} "
                f"(http://{self.device_id}.local:{self.port})"
            )
        except ImportError:
            logger.debug("zeroconf not installed, mDNS discovery disabled")
        except Exception as e:
            logger.warning(f"mDNS registration failed (non-fatal): {e}")

    def unregister(self) -> None:
        """注销 mDNS 服务。"""
        if self._service_info and self._zeroconf:
            try:
                self._zeroconf.unregister_service(self._service_info)
                self._zeroconf.close()
                logger.info("mDNS unregistered")
            except Exception as e:
                logger.warning(f"mDNS unregistration failed: {e}")
            finally:
                self._zeroconf = None
                self._service_info = None
