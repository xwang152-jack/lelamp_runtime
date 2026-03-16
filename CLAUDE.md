# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
使用中文交流。

## Project Overview

LeLamp Runtime is a Python-based control system for the LeLamp robotic lamp. It provides a conversational AI agent with motor control, RGB LED lighting, voice interaction, camera vision, and expression capabilities. The system runs on Raspberry Pi with servo motor hardware and integrates multiple AI services (DeepSeek LLM, Qwen VL, Baidu Speech).

## Development Environment

### Requirements

- **Python**: 3.12+ (required)
- **Package Manager**: UV (recommended)
- **Hardware**: Raspberry Pi with Feetech servo motors
- **OS**: Linux (Raspberry Pi OS) or macOS (for development without hardware)

### Package Management

This project uses **UV** as the package manager:

```bash
# Install dependencies (development machine - motor control only)
uv sync

# Install dependencies (Raspberry Pi - includes hardware-specific packages)
uv sync --extra hardware

# Install with vision support
uv sync --extra vision
```

**Important**: For LFS (Git Large File Storage) issues, use:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### Running Tests

```bash
# Find servo driver port
uv run lerobot-find-port

# Test RGB LEDs (requires sudo)
sudo uv run -m lelamp.test.test_rgb

# Test audio system
uv run -m lelamp.test.test_audio

# Test motors
uv run -m lelamp.test.test_motors --id <lamp_id> --port <port>
```

### Common Development Commands

```bash
# List available motor recordings
uv run -m lelamp.list_recordings --id <lamp_id>

# Record motor movements
uv run -m lelamp.record --id <lamp_id> --port <port> --name <recording_name>

# Replay motor movements
uv run -m lelamp.replay --id <lamp_id> --port <port> --name <recording_name>

# Calibrate motors
sudo uv run -m lelamp.calibrate --id <lamp_id> --port <port>

# Setup motors with unique IDs
uv run -m lelamp.setup_motors --id <lamp_id> --port <port>

# Run the main voice agent
sudo uv run main.py console
```

## Architecture

### Configuration Management

Centralized configuration system in `lelamp/config.py`:
- `AppConfig`: Main application configuration (LLM, Vision, Speech, Hardware)
- `MotorConfig`: Motor control settings
- `RGBConfig`: LED matrix configuration (brightness, pin, layout)
- `VisionConfig`: Camera settings including privacy protection
- `load_config()`: Loads all environment variables with type-safe defaults
- All configuration is frozen (immutable) to prevent runtime mutations

### Service-Based Architecture

The system uses a priority-based event dispatch architecture built on `ServiceBase`:

- **Priority System**: CRITICAL(0) > HIGH(1) > NORMAL(2) > LOW(3)
- **Event Dispatch**: Services dispatch events with priorities; higher-priority events preempt lower ones
- **Threading Model**: Each service runs in its own daemon thread
  - Services use `threading.Lock` for queue synchronization (NOT `asyncio.Lock`)
  - Cross-thread state requires `threading.Lock`, not asyncio locks
  - Event loops run within worker threads, not the main thread
- **Queue Implementation**: Uses `heapq`-based priority queue with configurable max size
  - Prevents event loss during high-load scenarios
  - Higher-priority events can replace lower-priority ones when queue is full
  - Tracks statistics: events dispatched, processed, and dropped
- **Key Services**:
  - `MotorsService`: Controls servo motors, plays recorded animations
  - `RGBService`: Manages 8x8 LED matrix with effects and patterns
  - `VisionService`: Captures camera frames for vision processing with privacy protection

**Privacy Protection** (`lelamp/service/vision/privacy.py`):
- `CameraPrivacyManager`: Manages camera privacy with LED indicators and user consent
- Camera states: IDLE, ACTIVE, PAUSED, CONSENT_REQUIRED
- LED indicators: Off (idle), Red breathing (active), Yellow blinking (consent needed)
- User consent system with configurable timeout and TTL (1 hour default)
- Usage statistics tracking (session count, total usage time)
- Thread-safe state management with `threading.Lock`
- `PrivacyGuard` context manager for automatic camera activation/deactivation

### Concurrency & Threading Critical

**Cross-Thread Safety**: The agent runs in LiveKit's asyncio event loop, but services use threading.
- When accessing shared state between agent (asyncio) and services (threaded), use `threading.Lock`
- Example: `self._conversation_state_lock = threading.Lock()` in main.py
- Never use `asyncio.Lock` for state shared across threads - it won't work

**Async Subprocess Calls**: Use `asyncio.create_subprocess_exec` instead of `subprocess.run` to avoid blocking the event loop.

### Motor Control System

- **Recording Format**: CSV files in `lelamp/recordings/` with columns: `timestamp, base_yaw.pos, base_pitch.pos, elbow_pitch.pos, wrist_roll.pos, wrist_pitch.pos`
- **Available Joints**: `base_yaw`, `base_pitch`, `elbow_pitch`, `wrist_roll`, `wrist_pitch`
- **Follower/Leader Modes**: Configured in `lelamp/follower/` and `lelamp/leader/` directories
- **Animation Playback**: Uses 30fps interpolation from recorded CSV data

**Health Monitoring** (`lelamp/service/motors/health_monitor.py`):
- `MotorHealthMonitor`: Monitors motor temperature, voltage, load, and position errors
- Health states: HEALTHY, WARNING, CRITICAL, STALLED
- Configurable thresholds via environment variables (see Configuration section)
- Background health check thread runs at configurable intervals (default 5min)
- Automatic protective actions on critical states (stop playback on stall/critical)
- Health history tracking and statistics (warning/critical/stall counts)
- Thread-safe with `threading.Lock` for concurrent access
- Usage: `motors_service.get_motor_health_summary()` returns health status of all motors

### AI Integration Layer

Located in `lelamp/integrations/`:

- **Baidu Speech**: Custom STT/TTS implementation for LiveKit agents (`baidu_speech.py`)
  - Handles state callbacks for conversation states (listening, thinking, speaking)
  - Provides Chinese speech recognition and synthesis
  - Uses shared `BaiduAuth` class for OAuth token management

- **Qwen Vision**: Vision-language model client (`qwen_vl.py`)
  - Integrates with ModelScope API for visual understanding
  - Supports homework checking and visual Q&A

### Utilities & Caching

**Rate Limiting** (`lelamp/utils/rate_limiter.py`):
- Token bucket algorithm for API rate limiting
- Prevents API abuse and cost overruns
- Use `get_rate_limiter(name, rate, capacity)` to get/create limiters
- Example: Search API limited to 2 req/s, Vision API to 0.5 req/s

**Response Caching** (`lelamp/cache/cache_manager.py`):
- TTL-based caching for LLM responses to reduce redundant API calls
- `VisionCache`: Caches vision API responses (50 items, 10min TTL)
- `SearchCache`: Caches search results (100 items, 5min TTL)
- Automatic expiration and LRU eviction

**Error Handling** (`lelamp/integrations/exceptions.py`):
- Unified exception hierarchy for all integration errors
- Exception types: `AuthenticationError`, `RateLimitError`, `NetworkError`, `ValidationError`, `ServiceUnavailableError`, `TimeoutError`
- Automatic retry with exponential backoff via `@retry_on_error` decorator
- Fallback strategies: `SilentFallback`, `MessageFallback`, `CachedFallback`
- Error conversion utilities for LiveKit APIError and httpx exceptions
- All exceptions mark whether they're retryable and include provider context

**Security** (`lelamp/utils/security.py`):
- Device ID generation (CPU serial on Linux, MAC address fallback)
- HMAC-SHA256 based license key generation with environment variable secret
- `LELAMP_LICENSE_KEY` environment variable for device authorization
- `LELAMP_LICENSE_SECRET` environment variable for signing (production required)
- `LELAMP_DEV_MODE` to bypass license checks in development

**URL Validation** (`lelamp/utils/url_validation.py`):
- SSRF protection for external API calls
- HTTPS enforcement
- Domain whitelist validation
- Private IP address blocking
- DNS resolution safety checks

**OTA Updates** (`lelamp/utils/ota.py`):
- `OTAManager`: Handles over-the-air firmware updates
- Version checking with semantic versioning (requires `packaging` library)
- Secure download with mandatory SHA256 hash verification
- HTTPS enforcement with SSL certificate verification
- Automatic rollback on update failure
- Thread-safe update operations with locking
- Download progress reporting (with `httpx`)
- Configured via `LELAMP_OTA_URL` environment variable

### Main Agent (main.py)

The `LeLamp` class extends `Agent` from LiveKit:

- **Personality**: Sarcastic, clumsy robot lamp that responds in Chinese
- **Function Tools**: All agent capabilities exposed as `@function_tool` decorators:
  - Motor control: `play_recording`, `move_joint`, `get_joint_positions`
  - RGB effects: `set_rgb_solid`, `paint_rgb_pattern`, `rgb_effect_*`
  - Vision: `vision_answer`, `check_homework`, `capture_to_feishu`
  - System: `set_volume`, `web_search`
  - Commercial: `get_motor_health`, `tune_motor_pid`, `reset_motor_health_stats`, `check_for_updates`, `perform_ota_update`

- **Conversation States**:
  - `idle`: Warm white light (255, 244, 229)
  - `listening`: Blue light (0, 140, 255)
  - `thinking`: Purple light (180, 0, 255)
  - `speaking`: Random animated colors with breathing effect

- **Cooldown & Override System**:
  - Motion cooldown prevents excessive movement (`LELAMP_MOTION_COOLDOWN_S`)
  - Light override system prevents state changes from overriding manual light commands
  - Motion suppression after light commands (`LELAMP_SUPPRESS_MOTION_AFTER_LIGHT_S`)

- **Input Validation**: All `move_joint` calls validate against `SAFE_JOINT_RANGES` to prevent mechanical damage:
  ```python
  SAFE_JOINT_RANGES = {
      "base_yaw": (-180, 180),
      "base_pitch": (-90, 90),
      "elbow_pitch": (-150, 150),
      "wrist_roll": (-180, 180),
      "wrist_pitch": (-90, 90),
  }
  ```

## Configuration

### Environment Variables

Required in `.env` file (create from template):

**LiveKit** (voice infrastructure):
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

**LLM** (conversational AI):
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL` (default: "deepseek-chat")
- `DEEPSEEK_BASE_URL` (default: "https://api.deepseek.com")

**Vision** (optional):
- `LELAMP_VISION_ENABLED` (default: true)
- `MODELSCOPE_API_KEY`
- `MODELSCOPE_MODEL` (default: "Qwen/Qwen3-VL-235B-A22B-Instruct")
- `MODELSCOPE_BASE_URL` (default: "https://api-inference.modelscope.cn/v1")
- `MODELSCOPE_TIMEOUT_S` (default: 60.0)
- `LELAMP_CAMERA_INDEX_OR_PATH` (default: "0")
- `LELAMP_CAMERA_WIDTH` (default: 1024)
- `LELAMP_CAMERA_HEIGHT` (default: 768)
- `LELAMP_CAMERA_ROTATE_DEG` (default: 0)
- `LELAMP_CAMERA_FLIP` (default: "none")
- `LELAMP_VISION_CAPTURE_INTERVAL_S` (default: 2.5)
- `LELAMP_VISION_JPEG_QUALITY` (default: 92)
- `LELAMP_VISION_MAX_AGE_S` (default: 15.0)

**Speech** (Baidu):
- `BAIDU_SPEECH_API_KEY`
- `BAIDU_SPEECH_SECRET_KEY`
- `BAIDU_SPEECH_CUID` (default: "lelamp")
- `BAIDU_SPEECH_TTS_PER` (default: 4)

**Hardware**:
- `LELAMP_PORT` (default: "/dev/ttyACM0")
- `LELAMP_ID` (default: "lelamp")

**Motor Health Monitoring** (Commercial Features):
- `LELAMP_MOTOR_HEALTH_CHECK_ENABLED` (default: true)
- `LELAMP_MOTOR_HEALTH_CHECK_INTERVAL_S` (default: 300.0, 5 minutes)
- `LELAMP_MOTOR_TEMP_WARNING_C` (default: 65.0, temperature warning threshold in Celsius)
- `LELAMP_MOTOR_TEMP_CRITICAL_C` (default: 75.0, temperature critical threshold)
- `LELAMP_MOTOR_VOLTAGE_MIN_V` (default: 11.0, minimum safe voltage)
- `LELAMP_MOTOR_VOLTAGE_MAX_V` (default: 13.0, maximum safe voltage)
- `LELAMP_MOTOR_LOAD_WARNING` (default: 0.8, load warning threshold 0-1)
- `LELAMP_MOTOR_LOAD_STALL` (default: 0.95, stall detection threshold 0-1)
- `LELAMP_MOTOR_POSITION_ERROR_DEG` (default: 5.0, position error tolerance in degrees)

**LED Matrix Configuration**:
- `LELAMP_LED_BRIGHTNESS` (default: 25)
- `LELAMP_MATRIX_W` (default: 8)
- `LELAMP_MATRIX_H` (default: 8)
- `LELAMP_MATRIX_LAYOUT` (default: "serpentine")
- `LELAMP_MATRIX_ORIGIN` (default: "top_left")
- `LELAMP_MATRIX_ROTATE_DEG` (default: 180)
- `LELAMP_MATRIX_FPS` (default: 30)

**Optional Features**:
- `BOCHA_API_KEY` (for web search)
- `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, `FEISHU_RECEIVE_ID` (for photo push)
- `LELAMP_GREETING_TEXT` (startup message)
- `LELAMP_BOOT_ANIMATION` (default: 1)
- `LELAMP_NOISE_CANCELLATION` (default: true)
- `LELAMP_STT_INPUT_GAIN` (default: 3.0)
- `LOG_LEVEL` (default: "INFO")

**Commercialization & OTA**:
- `LELAMP_LICENSE_KEY` (device authorization, see `scripts/generate_client_token.py`)
- `LELAMP_LICENSE_SECRET` (签名密钥，生产环境必需，务必保密)
- `LELAMP_DEV_MODE` (开发模式，设置为 1 跳过授权检查)
- `LELAMP_OTA_URL` (OTA update server endpoint)

⚠️ **Security**:
- Never commit `.env` file. Use `.env.example` as template.
- API keys in git history are security vulnerabilities.
- `LELAMP_LICENSE_SECRET` must be a strong random string (生产环境必须使用强随机密钥)
- All external API URLs are validated against domain whitelist to prevent SSRF attacks

### VAD Configuration

Silero VAD can be customized via environment variables:
- `LELAMP_VAD_MIN_SPEECH_DURATION`
- `LELAMP_VAD_MIN_SILENCE_DURATION`
- `LELAMP_VAD_PREFIX_PADDING_DURATION`
- `LELAMP_VAD_ACTIVATION_THRESHOLD`

## Code Organization

- `main.py`: Main entry point, agent definition, LiveKit integration
- `lelamp/config.py`: Centralized configuration management with type-safe loading
- `lelamp/service/`: Service architecture (base, motors, RGB, vision, privacy)
- `lelamp/integrations/`: External AI service clients (Baidu, Qwen VL) with unified error handling
- `lelamp/utils/`: Rate limiting, security utilities, OTA updates, shared helpers
- `lelamp/cache/`: TTL caching for LLM responses
- `lelamp/follower/` & `lelamp/leader/`: Motor control configurations
- `lelamp/recordings/`: CSV files with motor animation sequences
- `lelamp/test/`: Hardware testing utilities
- `web_client/`: Web-based user client for remote control and monitoring
- `scripts/`: Build and token generation utilities for commercial deployment
- `VERSION`: Current runtime version string (used for OTA updates)

## Hardware-Specific Notes

- **RGB LEDs**: Controlled via `rpi-ws281x` library (GPIO pin 12, 64 LEDs in 8x8 matrix)
  - Supports configurable brightness, layout (serpentine), origin, and rotation
  - Privacy indicator: LED turns red when camera is active
- **Motors**: Uses Feetech servo SDK with serial communication
  - 5 joints: base_yaw, base_pitch, elbow_pitch, wrist_roll, wrist_pitch
  - All movements validated against SAFE_JOINT_RANGES to prevent damage
- **Camera**: Supports rotation/flip via environment variables
  - Privacy protection with LED indicator when capturing
  - Configurable capture interval, JPEG quality, and frame caching
- **System Volume**: Controlled via `amixer` for Line/Line DAC/HP outputs

## Development Workflow

When adding new motor animations:
1. Record movement with `uv run -m lelamp.record --id <lamp_id> --port <port> --name <name>`
2. CSV file saved to `lelamp/recordings/<name>.csv`
3. Call via agent tool: `play_recording("<name>")`
4. Recordings are cached in memory after first load for 10x faster replay

When modifying agent behavior:
- Use `lelamp/config.py` to add new configuration variables with type-safe loading
- Edit the `instructions` parameter in `LeLamp.__init__()` for personality changes
- Add new `@function_tool` methods for new capabilities
- Services communicate via `dispatch(event_type, payload, priority)`
- Camera privacy: VisionService automatically controls privacy LED when capturing

When adding API integrations:
- Use unified error handling from `lelamp/integrations/exceptions.py`
- Apply `@retry_on_error` decorator for resilient API calls
- Implement fallback strategies with `@with_fallback` for graceful degradation
- Convert third-party exceptions to `IntegrationError` subclasses
- Use rate limiters from `lelamp/utils/rate_limiter.py` to prevent API abuse

When working with commercial features:
- Generate client tokens using `scripts/generate_client_token.py` for authentication
- Test web client locally before deploying to production
- Use `LELAMP_LICENSE_KEY` for device authorization in commercial deployments
- Implement OTA updates using `OTAManager` class for remote firmware updates
- Build distribution packages with `scripts/build_dist.sh` for releases

When debugging services:
- Check logs with `LOG_LEVEL=DEBUG`
- Service state available via `is_running` and `has_pending_event` properties
- Use `wait_until_idle(timeout)` before service shutdown

## Important Implementation Notes

### Threading Concurrency
- **Lock Types**: Use `threading.Lock` for cross-thread state, `asyncio.Lock` for asyncio-only state
- **Subprocess**: Always use `asyncio.create_subprocess_exec()` instead of `subprocess.run()` to avoid blocking
- **Shared State**: The agent (asyncio) and services (threaded) share state - use threading primitives
- **Critical Section**: All timestamp accesses (`_light_override_until_ts`, `_suppress_motion_until_ts`, `_last_motion_ts`) must be protected by `_timestamps_lock`

### API Safety
- **Rate Limiting**: Always use rate limiters for external API calls to prevent cost overruns
- **Caching**: Use vision/search caches to reduce redundant API calls by 30-50%
- **Input Validation**: Motor commands must validate against `SAFE_JOINT_RANGES`
- **Error Handling**: Use `@retry_on_error` decorator for resilient API calls with exponential backoff
- **Fallback Strategies**: Implement `@with_fallback` for graceful degradation when services fail
- **URL Validation**: All external URLs validated against whitelist to prevent SSRF attacks

### Security
- **Environment Variables**: Never commit `.env` file - use `.env.example` template
- **Git History**: API keys in git history are security vulnerabilities
- **Token Management**: Use shared `BaiduAuth` for OAuth tokens to avoid code duplication
- **Device Authorization**: Use `LELAMP_LICENSE_KEY` for device authentication (required in production)
- **License Secret**: `LELAMP_LICENSE_SECRET` must be strong random string (production required)
- **Camera Privacy**: Always use `CameraPrivacyManager` for camera access with user consent and LED indicators
- **HTTPS Only**: OTA updates and external API calls enforce HTTPS with certificate verification
- **Hash Verification**: OTA updates require mandatory SHA256 verification
- **SSRF Protection**: External URLs validated against domain whitelist

### Commercial Features

**Web Client** (`web_client/`):
- Browser-based user interface for remote control and monitoring
- Real-time video streaming via LiveKit WebRTC
- Bidirectional audio communication
- Control panel for lamp functions (lights, motors, vision)
- Token-based authentication for secure connections

**OTA Updates**:
- `OTAManager` class handles firmware updates remotely
- Version checking via `LELAMP_OTA_URL` endpoint
- SHA256 hash verification for secure downloads
- Thread-safe update process with locking
- Update progress reporting and error handling

**Token Generation** (`scripts/generate_client_token.py`):
- Generate LiveKit access tokens for client authentication
- Supports customizable room names and participant identities
- Used by web client and mobile apps for secure device access

**Build System** (`scripts/build_dist.sh`):
- Distribution build script for packaging releases
- Prepares runtime for OTA update distribution

### Performance
- **CSV Recordings**: Cached in memory after first load (10x speedup)
- **Event Queues**: Priority-based with configurable max size to prevent event loss
- **TTL Cache**: Auto-expires and uses LRU eviction
