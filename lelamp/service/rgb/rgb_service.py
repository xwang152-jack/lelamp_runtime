import math
import threading
import time
from typing import Any, List, Union
from rpi_ws281x import PixelStrip, Color
from ..base import ServiceBase


class RGBService(ServiceBase):
    def __init__(self, 
                 led_count: int = 64,
                 led_pin: int = 12,
                 led_freq_hz: int = 800000,
                 led_dma: int = 10,
                 led_brightness: int = 255,
                 led_invert: bool = False,
                 led_channel: int = 0):
        super().__init__("rgb")
        
        self.led_count = led_count
        self._brightness = max(0, min(255, int(led_brightness)))
        self.strip = PixelStrip(
            led_count, led_pin, led_freq_hz, led_dma, 
            led_invert, self._brightness, led_channel
        )
        self.strip.begin()

        self._strip_lock = threading.Lock()
        self._breath_stop = threading.Event()
        self._breath_enabled = threading.Event()
        self._breath_rgb = (255, 255, 255)
        self._breath_period_s = 1.6
        self._breath_min = 10
        self._breath_max = 255
        self._breath_t0 = time.time()
        self._breath_thread = threading.Thread(target=self._breath_loop, daemon=True)
        self._breath_thread.start()
        
    def handle_event(self, event_type: str, payload: Any):
        if event_type == "solid":
            self._disable_breath()
            self._handle_solid(payload)
        elif event_type == "paint":
            self._disable_breath()
            self._handle_paint(payload)
        elif event_type == "breath":
            self._handle_breath(payload)
        elif event_type == "brightness":
            self._handle_brightness(payload)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _handle_solid(self, color_code: Union[int, tuple]):
        """Fill entire strip with single color"""
        with self._strip_lock:
            if isinstance(color_code, tuple) and len(color_code) == 3:
                color = Color(color_code[0], color_code[1], color_code[2])
            elif isinstance(color_code, int):
                color = color_code
            else:
                self.logger.error(f"Invalid color format: {color_code}")
                return

            self.strip.setBrightness(int(self._brightness))
            for i in range(self.led_count):
                self.strip.setPixelColor(i, color)
            self.strip.show()
        self.logger.debug(f"Applied solid color: {color_code}")
    
    def _handle_paint(self, colors: List[Union[int, tuple]]):
        """Set individual pixel colors from array"""
        if not isinstance(colors, list):
            self.logger.error(f"Paint payload must be a list, got: {type(colors)}")
            return
            
        max_pixels = min(len(colors), self.led_count)

        with self._strip_lock:
            self.strip.setBrightness(int(self._brightness))
            for i in range(max_pixels):
                color_code = colors[i]
                if isinstance(color_code, tuple) and len(color_code) == 3:
                    color = Color(color_code[0], color_code[1], color_code[2])
                elif isinstance(color_code, int):
                    color = color_code
                else:
                    self.logger.warning(f"Invalid color at index {i}: {color_code}")
                    continue

                self.strip.setPixelColor(i, color)

            self.strip.show()
        self.logger.debug(f"Applied paint pattern with {max_pixels} colors")

    def _handle_brightness(self, payload: Any):
        v = None
        if isinstance(payload, int):
            v = payload
        elif isinstance(payload, dict):
            if "value" in payload:
                v = payload.get("value")
            elif "brightness" in payload:
                v = payload.get("brightness")
            elif "percent" in payload:
                try:
                    p = float(payload.get("percent"))
                except Exception:
                    p = None
                if p is not None:
                    v = int(round(max(0.0, min(100.0, p)) * 255.0 / 100.0))

        if not isinstance(v, int):
            self.logger.error(f"Invalid brightness payload: {payload}")
            return

        self._brightness = max(0, min(255, int(v)))
        with self._strip_lock:
            self.strip.setBrightness(int(self._brightness))
            self.strip.show()

    def _handle_breath(self, payload: Any):
        rgb = None
        period_s = None
        min_brightness = None
        max_brightness = None

        if isinstance(payload, dict):
            v = payload.get("rgb")
            if isinstance(v, (list, tuple)) and len(v) == 3:
                rgb = (int(v[0]), int(v[1]), int(v[2]))
            period_s = payload.get("period_s")
            min_brightness = payload.get("min_brightness")
            max_brightness = payload.get("max_brightness")
        elif isinstance(payload, (list, tuple)) and len(payload) == 3:
            rgb = (int(payload[0]), int(payload[1]), int(payload[2]))

        if rgb is None:
            self.logger.error(f"Invalid breath payload: {payload}")
            return

        if isinstance(period_s, (int, float)) and float(period_s) > 0:
            self._breath_period_s = float(period_s)
        if isinstance(min_brightness, int):
            self._breath_min = max(0, min(255, int(min_brightness)))
        if isinstance(max_brightness, int):
            self._breath_max = max(0, min(255, int(max_brightness)))
        if self._breath_max < self._breath_min:
            self._breath_min, self._breath_max = self._breath_max, self._breath_min

        self._breath_rgb = rgb
        self._breath_t0 = time.time()
        with self._strip_lock:
            c = Color(rgb[0], rgb[1], rgb[2])
            for i in range(self.led_count):
                self.strip.setPixelColor(i, c)
            self.strip.show()
        self._breath_enabled.set()

    def _disable_breath(self):
        self._breath_enabled.clear()

    def _breath_loop(self):
        while not self._breath_stop.is_set():
            if not self._breath_enabled.is_set():
                time.sleep(0.05)
                continue

            now = time.time()
            period = float(self._breath_period_s) if self._breath_period_s else 1.6
            phase = ((now - self._breath_t0) / period) * (2.0 * math.pi)
            v = (math.sin(phase - math.pi / 2.0) + 1.0) / 2.0
            brightness = int(self._breath_min + v * (self._breath_max - self._breath_min))
            brightness = min(int(self._brightness), max(0, brightness))

            with self._strip_lock:
                self.strip.setBrightness(brightness)
                self.strip.show()

            time.sleep(1.0 / 30.0)
    
    def clear(self):
        """Turn off all LEDs"""
        self._disable_breath()
        with self._strip_lock:
            self.strip.setBrightness(int(self._brightness))
            for i in range(self.led_count):
                self.strip.setPixelColor(i, Color(0, 0, 0))
            self.strip.show()
    
    def stop(self, timeout: float = 5.0):
        """Override stop to clear LEDs before stopping"""
        self.clear()
        self._breath_stop.set()
        if self._breath_thread and self._breath_thread.is_alive():
            self._breath_thread.join(timeout=timeout)
        super().stop(timeout)
