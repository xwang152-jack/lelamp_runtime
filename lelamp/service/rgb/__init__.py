try:
    from .rgb_service import RGBService
except ImportError:
    pass
from .noop_rgb_service import NoOpRGBService

__all__ = ["RGBService", "NoOpRGBService"]
