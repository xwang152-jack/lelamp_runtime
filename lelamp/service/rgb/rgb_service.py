import os
import math
import random
import threading
import time
from typing import Any, Dict, List, Union
from rpi_ws281x import PixelStrip, Color
from ..base import ServiceBase


class RGBService(ServiceBase):
    def __init__(self, 
                 led_count: int = 64,
                 led_pin: int = 12,
                 led_freq_hz: int = 800000,
                 led_dma: int = 10,
                 led_brightness: int = 25,
                 led_invert: bool = False,
                 led_channel: int = 0):
        super().__init__("rgb")
        
        self.led_count = led_count
        # 支持通过环境变量覆盖默认亮度
        env_brightness = os.getenv("LELAMP_LED_BRIGHTNESS")
        if env_brightness:
            try:
                led_brightness = int(env_brightness)
            except ValueError:
                pass
        
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

        mw, mh = self._infer_matrix_size(self.led_count)
        self._matrix_w = int(os.getenv("LELAMP_MATRIX_W") or str(mw))
        self._matrix_h = int(os.getenv("LELAMP_MATRIX_H") or str(mh))
        self._matrix_layout = (os.getenv("LELAMP_MATRIX_LAYOUT") or "serpentine").strip().lower()
        self._matrix_origin = (os.getenv("LELAMP_MATRIX_ORIGIN") or "top_left").strip().lower()
        self._matrix_rotate = int(os.getenv("LELAMP_MATRIX_ROTATE_DEG") or "180")

        self._effect_stop = threading.Event()
        self._effect_enabled = threading.Event()
        self._effect_cfg: Dict[str, Any] = {}
        self._effect_t0 = time.time()
        self._effect_thread = threading.Thread(target=self._effect_loop, daemon=True)
        self._effect_thread.start()
        self._fire_heat: List[List[int]] | None = None
        
    def handle_event(self, event_type: str, payload: Any):
        if event_type == "solid":
            self._disable_breath()
            self._disable_effect()
            self._handle_solid(payload)
        elif event_type == "paint":
            self._disable_breath()
            self._disable_effect()
            self._handle_paint(payload)
        elif event_type == "breath":
            self._disable_effect()
            self._handle_breath(payload)
        elif event_type == "brightness":
            self._handle_brightness(payload)
        elif event_type == "effect":
            self._disable_breath()
            self._handle_effect(payload)
        elif event_type == "effect_stop":
            self._disable_effect()
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

    def _disable_effect(self):
        self._effect_enabled.clear()
        self._effect_cfg = {}
        self._fire_heat = None

    def _handle_effect(self, payload: Any):
        if not isinstance(payload, dict):
            self.logger.error(f"Invalid effect payload: {payload}")
            return
        name = str(payload.get("name") or "").strip().lower()
        if not name:
            self.logger.error(f"Invalid effect payload: {payload}")
            return
        fps = payload.get("fps")
        if isinstance(fps, (int, float)) and float(fps) > 0:
            fps = float(fps)
        else:
            fps = float(os.getenv("LELAMP_MATRIX_FPS") or "30")
        self._effect_cfg = dict(payload)
        self._effect_cfg["name"] = name
        self._effect_cfg["fps"] = fps
        self._effect_t0 = time.time()
        if name == "fire":
            self._fire_heat = [[0 for _ in range(self._matrix_w)] for _ in range(self._matrix_h)]
        self._effect_enabled.set()

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

    def _effect_loop(self):
        while not self._effect_stop.is_set():
            if not self._effect_enabled.is_set():
                time.sleep(0.05)
                continue

            cfg = self._effect_cfg
            name = str(cfg.get("name") or "")
            fps = float(cfg.get("fps") or 30.0)
            if fps <= 0:
                fps = 30.0
            t = time.time()
            frame = self._render_effect(name, cfg, t)
            if frame:
                with self._strip_lock:
                    self.strip.setBrightness(int(self._brightness))
                    for i, rgb in enumerate(frame[: self.led_count]):
                        if isinstance(rgb, int):
                            color = rgb
                        else:
                            r, g, b = rgb
                            color = Color(int(r), int(g), int(b))
                        self.strip.setPixelColor(i, color)
                    self.strip.show()
            time.sleep(1.0 / fps)

    def _render_effect(self, name: str, cfg: Dict[str, Any], t: float) -> List[Union[int, tuple]]:
        if name == "rainbow":
            return self._render_rainbow(cfg, t)
        if name == "wave":
            return self._render_wave(cfg, t)
        if name == "fire":
            return self._render_fire(cfg, t)
        if name in ("emoji", "emote", "face"):
            return self._render_emoji(cfg, t)
        return []

    def _render_rainbow(self, cfg: Dict[str, Any], t: float) -> List[tuple]:
        speed = float(cfg.get("speed") or 1.0)
        sat = float(cfg.get("saturation") or 1.0)
        val = float(cfg.get("value") or 1.0)
        sat = max(0.0, min(1.0, sat))
        val = max(0.0, min(1.0, val))
        out: List[tuple] = [(0, 0, 0) for _ in range(self.led_count)]
        for y in range(self._matrix_h):
            for x in range(self._matrix_w):
                h = ((x + y) / max(1.0, (self._matrix_w + self._matrix_h - 2))) + (t - self._effect_t0) * 0.12 * speed
                r, g, b = self._hsv_to_rgb(h % 1.0, sat, val)
                idx = self._xy_to_index(x, y)
                if 0 <= idx < self.led_count:
                    out[idx] = (r, g, b)
        return out

    def _render_wave(self, cfg: Dict[str, Any], t: float) -> List[tuple]:
        speed = float(cfg.get("speed") or 1.0)
        freq = float(cfg.get("freq") or 1.2)
        base = cfg.get("color") or cfg.get("rgb") or (60, 180, 255)
        if isinstance(base, (list, tuple)) and len(base) == 3:
            br, bg, bb = int(base[0]), int(base[1]), int(base[2])
        else:
            br, bg, bb = 60, 180, 255
        out: List[tuple] = [(0, 0, 0) for _ in range(self.led_count)]
        tt = (t - self._effect_t0) * speed
        for y in range(self._matrix_h):
            for x in range(self._matrix_w):
                phase = (x / max(1.0, (self._matrix_w - 1))) * (2.0 * math.pi * freq) + tt * 2.0 * math.pi
                v = (math.sin(phase + (y * 0.35)) + 1.0) / 2.0
                r = int(br * (0.15 + 0.85 * v))
                g = int(bg * (0.15 + 0.85 * v))
                b = int(bb * (0.15 + 0.85 * v))
                idx = self._xy_to_index(x, y)
                if 0 <= idx < self.led_count:
                    out[idx] = (r, g, b)
        return out

    def _render_fire(self, cfg: Dict[str, Any], t: float) -> List[tuple]:
        intensity = float(cfg.get("intensity") or 1.0)
        cooling = int(55 + (1.0 - max(0.0, min(1.0, intensity))) * 55)
        sparking = int(60 + max(0.0, min(1.0, intensity)) * 120)
        if self._fire_heat is None or len(self._fire_heat) != self._matrix_h or len(self._fire_heat[0]) != self._matrix_w:
            self._fire_heat = [[0 for _ in range(self._matrix_w)] for _ in range(self._matrix_h)]

        for x in range(self._matrix_w):
            for y in range(self._matrix_h):
                self._fire_heat[y][x] = max(0, self._fire_heat[y][x] - random.randint(0, cooling))

        for x in range(self._matrix_w):
            for y in range(self._matrix_h - 1, 1, -1):
                self._fire_heat[y][x] = int(
                    (self._fire_heat[y - 1][x] + self._fire_heat[y - 2][x] + self._fire_heat[y - 2][x]) / 3
                )

        for x in range(self._matrix_w):
            if random.randint(0, 255) < sparking:
                y = 0
                self._fire_heat[y][x] = min(255, self._fire_heat[y][x] + random.randint(160, 255))

        out: List[tuple] = [(0, 0, 0) for _ in range(self.led_count)]
        for y in range(self._matrix_h):
            for x in range(self._matrix_w):
                heat = int(self._fire_heat[y][x])
                r, g, b = self._heat_color(heat)
                idx = self._xy_to_index(x, self._matrix_h - 1 - y)
                if 0 <= idx < self.led_count:
                    out[idx] = (r, g, b)
        return out

    def _render_emoji(self, cfg: Dict[str, Any], t: float) -> List[tuple]:
        name = str(cfg.get("emoji") or cfg.get("name_emoji") or cfg.get("face") or cfg.get("face_name") or "smile").strip().lower()
        fg = cfg.get("color") or cfg.get("rgb") or (255, 200, 60)
        bg = cfg.get("bg") or (0, 0, 0)
        if isinstance(fg, (list, tuple)) and len(fg) == 3:
            fr, fg_, fb = int(fg[0]), int(fg[1]), int(fg[2])
        else:
            fr, fg_, fb = 255, 200, 60
        if isinstance(bg, (list, tuple)) and len(bg) == 3:
            br, bg_, bb = int(bg[0]), int(bg[1]), int(bg[2])
        else:
            br, bg_, bb = 0, 0, 0
        blink = bool(cfg.get("blink", True))
        period = float(cfg.get("period_s") or 2.2)
        frame = self._emoji_frame(name, blink, period, t - self._effect_t0)
        out: List[tuple] = [(br, bg_, bb) for _ in range(self.led_count)]
        for y in range(min(self._matrix_h, 8)):
            row = frame[y]
            for x in range(min(self._matrix_w, 8)):
                if (row >> (7 - x)) & 1:
                    idx = self._xy_to_index(x, y)
                    if 0 <= idx < self.led_count:
                        out[idx] = (fr, fg_, fb)
        return out

    def _emoji_frame(self, name: str, blink: bool, period_s: float, dt: float) -> List[int]:
        frames = self._emoji_frames().get(name) or self._emoji_frames().get("smile")
        if not frames:
            return [0 for _ in range(8)]
        if len(frames) == 1:
            return frames[0]
        if not blink:
            return frames[0]
        if period_s <= 0:
            period_s = 2.2
        phase = dt % period_s
        if phase > period_s * 0.86:
            return frames[1]
        return frames[0]

    def _emoji_frames(self) -> Dict[str, List[List[int]]]:
        return {
            "smile": [
                [
                    0b00111100,
                    0b01000010,
                    0b10100101,
                    0b10000001,
                    0b10100101,
                    0b10011001,
                    0b01000010,
                    0b00111100,
                ],
                [
                    0b00111100,
                    0b01000010,
                    0b10111101,
                    0b10000001,
                    0b10111101,
                    0b10011001,
                    0b01000010,
                    0b00111100,
                ],
            ],
            "sad": [
                [
                    0b00111100,
                    0b01000010,
                    0b10100101,
                    0b10000001,
                    0b10011001,
                    0b10100101,
                    0b01000010,
                    0b00111100,
                ],
                [
                    0b00111100,
                    0b01000010,
                    0b10111101,
                    0b10000001,
                    0b10011001,
                    0b10100101,
                    0b01000010,
                    0b00111100,
                ],
            ],
            "wink": [
                [
                    0b00111100,
                    0b01000010,
                    0b10100101,
                    0b10000001,
                    0b10100101,
                    0b10011001,
                    0b01000010,
                    0b00111100,
                ],
                [
                    0b00111100,
                    0b01000010,
                    0b10100101,
                    0b10000001,
                    0b10111101,
                    0b10011001,
                    0b01000010,
                    0b00111100,
                ],
            ],
            "angry": [
                [
                    0b00111100,
                    0b01000010,
                    0b10011001,
                    0b10100101,
                    0b10000001,
                    0b10100101,
                    0b01000010,
                    0b00111100,
                ],
                [
                    0b00111100,
                    0b01000010,
                    0b10011001,
                    0b10111101,
                    0b10000001,
                    0b10111101,
                    0b01000010,
                    0b00111100,
                ],
            ],
            "heart": [
                [
                    0b00000000,
                    0b01100110,
                    0b11111111,
                    0b11111111,
                    0b11111111,
                    0b01111110,
                    0b00111100,
                    0b00011000,
                ],
                [
                    0b00000000,
                    0b01100110,
                    0b11111111,
                    0b11111111,
                    0b01111110,
                    0b00111100,
                    0b00011000,
                    0b00000000,
                ],
            ],
        }

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> tuple:
        h = h % 1.0
        s = max(0.0, min(1.0, s))
        v = max(0.0, min(1.0, v))
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return int(r * 255), int(g * 255), int(b * 255)

    def _heat_color(self, heat: int) -> tuple:
        heat = max(0, min(255, int(heat)))
        t192 = (heat * 192) // 255
        if t192 > 128:
            r = 255
            g = 255
            b = (t192 - 128) * 2
        elif t192 > 64:
            r = 255
            g = (t192 - 64) * 4
            b = 0
        else:
            r = t192 * 4
            g = 0
            b = 0
        return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

    def _infer_matrix_size(self, n: int) -> tuple[int, int]:
        if n <= 0:
            return 1, 1
        s = int(round(math.sqrt(n)))
        if s * s == n:
            return s, s
        return n, 1

    def _xy_to_index(self, x: int, y: int) -> int:
        x = int(x)
        y = int(y)
        if self._matrix_origin == "bottom_left":
            y = (self._matrix_h - 1) - y
        elif self._matrix_origin == "bottom_right":
            x = (self._matrix_w - 1) - x
            y = (self._matrix_h - 1) - y
        elif self._matrix_origin == "top_right":
            x = (self._matrix_w - 1) - x

        rot = int(self._matrix_rotate) % 360
        if rot == 90:
            x, y = (self._matrix_h - 1 - y), x
        elif rot == 180:
            x, y = (self._matrix_w - 1 - x), (self._matrix_h - 1 - y)
        elif rot == 270:
            x, y = y, (self._matrix_w - 1 - x)

        if x < 0 or y < 0 or x >= self._matrix_w or y >= self._matrix_h:
            return -1
        if self._matrix_layout == "serpentine" and (y % 2 == 1):
            x = (self._matrix_w - 1) - x
        return y * self._matrix_w + x
    
    def clear(self):
        """Turn off all LEDs"""
        self._disable_breath()
        self._disable_effect()
        with self._strip_lock:
            self.strip.setBrightness(int(self._brightness))
            for i in range(self.led_count):
                self.strip.setPixelColor(i, Color(0, 0, 0))
            self.strip.show()
    
    def stop(self, timeout: float = 5.0):
        """Override stop to clear LEDs before stopping"""
        self.clear()
        self._breath_stop.set()
        self._effect_stop.set()
        if self._breath_thread and self._breath_thread.is_alive():
            self._breath_thread.join(timeout=timeout)
        if self._effect_thread and self._effect_thread.is_alive():
            self._effect_thread.join(timeout=timeout)
        super().stop(timeout)
