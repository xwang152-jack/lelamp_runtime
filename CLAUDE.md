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

# Install with API support (FastAPI, database)
uv sync --extra api

# Install with development tools (pytest, ruff)
uv sync --extra dev

# Install all extras
uv sync --extra hardware --extra vision --extra api --extra dev
```

**Important**: For LFS (Git Large File Storage) issues, use:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

**Project Dependencies** (from `pyproject.toml`):
- **Core**: `feetech-servo-sdk`, `lerobot`, `livekit`, `livekit-agents`
- **Audio**: `pyaudio`, `sounddevice`, `pvporcupine`, `pvrecorder`
- **Vision**: `opencv-python`, `mediapipe` (platform-specific)
- **Hardware**: `rpi-ws281x`, `adafruit-circuitpython-neopixel` (Linux only)
- **API**: `fastapi`, `uvicorn`, `pydantic`, `sqlalchemy`, `aiosqlite`
- **Utilities**: `httpx`, `python-dotenv`, `numpy`, `packaging`

**Note**: The `lerobot` package is installed from GitHub source:
```toml
[tool.uv.sources]
lerobot = { git = "https://github.com/huggingface/lerobot" }
```

### Running Tests

```bash
# Find servo driver port
uv run lerobot-find-port

# Test RGB LEDs (requires sudo)
sudo uv run -m lelamp.test.hardware.test_rgb

# Test audio system
uv run -m lelamp.test.hardware.test_audio

# Test motors
uv run -m lelamp.test.hardware.test_motors --id <lamp_id> --port <port>

# Run all tests with coverage
uv run pytest tests/ --cov=lelamp --cov-report=html

# Run a single test file
uv run pytest tests/test_basic.py -v

# Run tests matching a pattern
uv run pytest tests/ -k "test_setup"
```

### Linting & Code Quality

```bash
# Run ruff linter
uv run ruff check lelamp/

# Run ruff with auto-fix
uv run ruff check --fix lelamp/

# Format code with ruff
uv run ruff format lelamp/
```

### API & Database Commands

```bash
# Start FastAPI server (development)
uv run uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --reload

# Start FastAPI server (production with workers)
.venv/bin/python -m uvicorn lelamp.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Initialize database
uv run python -c "
from lelamp.database.session import engine
from lelamp.database.models import Base
Base.metadata.create_all(bind=engine)
print('Database initialized')
"

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
# Health check: http://localhost:8000/health
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

### Project Structure Overview

```
lelamp_runtime/
├── main.py                      # Entry point for LiveKit agent
├── pyproject.toml              # UV package manager configuration
├── VERSION                     # Runtime version for OTA updates
├── models/                     # Edge AI models (.tflite, .task)
│
├── lelamp/                     # Core package
│   ├── agent/                  # LiveKit Agent architecture
│   │   ├── lelamp_agent.py     # Main LeLamp Agent class
│   │   ├── states.py           # Conversation state management
│   │   └── tools/              # Function tools (motor, rgb, vision, edge_vision, system)
│   ├── edge/                   # Edge AI inference (MediaPipe Tasks API)
│   │   ├── face_detector.py    # Face detection + presence
│   │   ├── hand_tracker.py     # Hand tracking + gesture recognition
│   │   ├── object_detector.py  # Object detection (COCO 80 classes)
│   │   └── hybrid_vision.py    # Smart routing: local vs cloud
│   ├── api/                    # FastAPI REST API & WebSocket
│   ├── database/               # SQLAlchemy ORM layer
│   ├── config.py               # Centralized configuration (frozen dataclasses)
│   ├── service/                # Priority-based event dispatch services
│   │   ├── base.py             # ServiceBase with heapq priority queue
│   │   ├── motors/             # Motor service + health monitoring
│   │   ├── rgb/                # RGB LED matrix service
│   │   └── vision/             # Camera capture + privacy + proactive monitor
│   ├── integrations/           # External AI clients (Baidu Speech, Qwen VL)
│   ├── utils/                  # Rate limiting, security, OTA
│   └── recordings/             # Motor animation CSV files
│
├── web/                        # Vue 3 frontend
└── scripts/                    # Build and deployment scripts
```

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
- Edge vision callbacks (gesture/presence) run in background threads — motor/RGB dispatches from callbacks are thread-safe

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

### Authentication System (`lelamp/api/` & `lelamp/database/`)

**JWT-Based Authentication**: Complete user authentication system with access and refresh tokens:

- **Access Token**: Short-lived (30 minutes) token for API authentication
- **Refresh Token**: Long-lived (7 days) token for obtaining new access tokens
- **Password Security**: bcrypt hashing with automatic salt generation
- **Token Storage**: Access tokens in client memory, refresh tokens in database
- **Token Refresh**: Automatic token rotation with old token revocation

**API Endpoints** (`lelamp/api/routes/auth.py`):
```python
POST /api/auth/register      # User registration
POST /api/auth/login         # User login
POST /api/auth/refresh-token # Token refresh
GET  /api/auth/me            # Get current user info
POST /api/auth/bind-device   # Bind device to user
```

**Database Models** (`lelamp/database/models_auth.py`):
```python
class User(Base):
    # User account with username, email, hashed_password
    # Indexes: username (unique), email (unique)

class RefreshToken(Base):
    # Refresh tokens with expiry tracking
    # Indexes: token (unique), user_id, expires_at

class DeviceBinding(Base):
    # User-to-device binding with permissions
    # Foreign key: User
```

**Authentication Service** (`lelamp/api/services/auth_service.py`):
```python
class AuthService:
    @staticmethod
    def register_user(db, username, email, password):
        # Create user with bcrypt password hash

    @staticmethod
    def authenticate_user(db, username, password):
        # Verify credentials and return user

    @staticmethod
    def create_access_token(data):
        # Generate JWT (30 min expiry)

    @staticmethod
    def create_refresh_token(user_id, db):
        # Generate JWT with UUID jti (7 days expiry)

    @staticmethod
    def verify_token(token, token_type):
        # Decode and validate JWT
```

**Authentication Middleware** (`lelamp/api/middleware/auth.py`):
```python
# Dependency injection for protected routes
async def get_current_user(token: str, db: Session) -> User:
    # Verify JWT, query database, return user

async def get_current_user_optional(token: str, db: Session) -> Optional[User]:
    # Optional authentication (allow anonymous)

async def get_current_admin(user: User) -> User:
    # Require admin role
```

### Middleware System (`lelamp/api/middleware/`)

**Rate Limiting** (`rate_limit.py`):
- **Algorithm**: Sliding window with configurable limits
- **Identification**: User ID (authenticated) or IP address (anonymous)
- **Limits**: default (100/min), strict (20/min), loose (1000/min)
- **Response**: 429 Too Many Requests with retry-after header

```python
from lelamp.api.middleware.rate_limit import RateLimiter, RateLimitDep

limiter = RateLimiter()

# Apply to endpoints
@router.get("/api/devices/{lamp_id}/state")
@RateLimitDep(max_requests=100, window_seconds=60)
async def get_device_state():
    # Endpoint logic
```

**API Caching** (`cache.py`):
- **Scope**: GET requests only
- **Storage**: In-memory dictionary with TTL
- **Cache Key**: MD5 hash of URL
- **Decorator**: `@cache_response(ttl_seconds=60)`

```python
from lelamp.api.middleware.cache import cache_response

@router.get("/api/devices/{lamp_id}/state")
@cache_response(ttl_seconds=30)
async def get_device_state(request: Request, lamp_id: str):
    # Data cached for 30 seconds
```

**Security Headers** (in `lelamp/api/app.py`):
- `X-Content-Type-Options: nosniff` - Prevent MIME sniffing
- `X-Frame-Options: DENY` - Prevent clickjacking
- `X-XSS-Protection: 1; mode=block` - Enable XSS protection
- `Strict-Transport-Security: max-age=31536000` - Force HTTPS
- `Content-Security-Policy: default-src 'self'` - Prevent XSS
- `Referrer-Policy: strict-origin-when-cross-origin` - Control referrer leaks
- `Permissions-Policy: geolocation=(), microphone=(), camera=()` - Restrict browser features

**WebSocket Authentication** (`lelamp/api/routes/websocket.py`):
```python
@router.websocket("/{lamp_id}")
async def websocket_endpoint(websocket: WebSocket, lamp_id: str, token: Optional[str] = Query(None)):
    # Optional JWT authentication via query parameter
    # Anonymous connections allowed
    # Invalid token logged as warning
    # Expired token rejects with 1008 error code
```

### Agent Architecture (`lelamp/agent/`)

**Modular Agent Design**: The agent logic has been refactored into separate modules:

- `lelamp_agent.py`: Main `LeLamp` class extending LiveKit's `Agent`
  - Personality: Sarcastic, clumsy robot lamp responding in Chinese
  - Conversation state management with LED color indicators
  - Cooldown and override systems for motion and lighting

- `states.py`: State management for conversation flow
  - `ConversationState`: Enum (IDLE, LISTENING, THINKING, SPEAKING)
  - `StateManager`: Thread-safe state transitions with callbacks
  - `StateColors`: Maps conversation states to RGB colors

- `tools/`: Function tools organized by domain
  - `motor_tools.py`: Motor control and health monitoring
  - `rgb_tools.py`: LED effects and color control
  - `vision_tools.py`: Camera capture and cloud vision AI (Qwen VL)
  - `edge_vision_tools.py`: Local edge vision (face, gesture, object)
  - `system_tools.py`: System utilities (volume, web search, OTA)

### Edge Vision Architecture (`lelamp/edge/`)

Local AI inference using **MediaPipe Tasks API** (0.10+) for low-latency, privacy-preserving visual processing.

**Components** (all use `mediapipe.tasks.vision` API):

| Component | API Class | Model File | Purpose |
|-----------|-----------|------------|---------|
| `FaceDetector` | `FaceDetector` | `blaze_face_full_range.tflite` | Presence detection with debouncing |
| `HandTracker` | `HandLandmarker` | `hand_landmarker.task` | 21-point hand tracking + gesture recognition |
| `ObjectDetector` | `ObjectDetector` | `efficientdet_lite0.tflite` | 80-class COCO object detection |
| `HybridVisionService` | Orchestrator | — | Routes queries: local vs cloud vs hybrid |

**Graceful Degradation**: All components check `hasattr(mp, 'tasks')` at import time. If unavailable, they run in NoOp mode returning empty results — the system never crashes due to missing MediaPipe.

**Hybrid Routing** (`hybrid_vision.py`):
- `SIMPLE` queries ("这是什么") → local `ObjectDetector` only
- `COMPLEX` queries ("检查作业") → cloud Qwen VL only
- `MODERATE` queries → try local first, fall back to cloud if confidence < threshold

**Proactive Vision Monitor** (`lelamp/service/vision/proactive_vision_monitor.py`):
- Background thread that continuously monitors camera for gestures and presence
- Activated by `LELAMP_PROACTIVE_MONITOR=true` (default) when edge vision is enabled
- Auto-adjusts FPS: ACTIVE (5 fps) when user present, IDLE (1 fps) when absent
- Triggers motor/LED callbacks on gesture recognition (thumbs up → nod, wave → toggle light)

**Gesture Recognition** (`hand_tracker.py`):
- Recognizes 8 gestures: OPEN, FIST, POINT, PEACE, THUMBS_UP, THUMBS_DOWN, OK, WAVE
- Wave detection uses temporal analysis (horizontal wrist movement over N frames)
- Per-gesture cooldown (default 1s) prevents repeated triggering
- Handedness-aware thumb detection (left vs right hand)

**Model Files** (in `models/` directory):
- `blaze_face_full_range.tflite` — Face detection
- `hand_landmarker.task` — Hand landmarks + tracking
- `efficientdet_lite0.tflite` — Object detection
- `gesture_recognizer.task` — Available but not used (gesture recognition is done via landmark analysis)

### API & Database Layer (`lelamp/api/` & `lelamp/database/`)

**RESTful API System**:
- FastAPI application with auto-generated OpenAPI docs (`/docs`, `/redoc`)
- 9 REST endpoints for device management, settings, and history
- WebSocket support for real-time state updates (`/api/ws/{lamp_id}`)
- CORS enabled for web client integration
- 13 WebSocket message types for event pushing

**Database Models** (`lelamp/database/models.py`):
- `Conversation`: Chat history with messages, duration, and AI responses
- `OperationLog`: System operation tracking with success/failure status
- `DeviceState`: Device state snapshots for monitoring
- `UserSettings`: User preferences and configuration storage

**Captive Portal** (`lelamp/api/services/captive_portal.py`):
- Self-contained FastAPI app with embedded HTML/CSS/JS for first-time device setup
- Provides WiFi scanning, connection, and setup completion flow
- Runs on separate port (8080) with its own systemd service
- Endpoints: `/api/setup/status`, `/api/setup/networks`, `/api/setup/connect`, `/api/setup/reset`
- Related scripts: `scripts/lelamp-captive-portal.service`, `scripts/install_captive_portal.sh`

**API Services** (`lelamp/api/services/`):
- `wifi_manager.py`: WiFi network scanning and connection management
- `ap_manager.py`: Access point mode for onboarding
- `config_sync.py`: Configuration synchronization between services
- `onboarding.py`: First-time setup wizard logic
- `captive_portal.py`: Web-based first-time setup portal with embedded HTML UI for WiFi configuration
- `setup_state.py`: Setup state persistence and recovery
- `wifi_scanner.py`: Async WiFi network scanning
- `network_manager.py`: Network connection management with retry logic

### Legacy Main Agent (main.py)

**Entry Point**: `main.py` now serves as the bootstrap for the LiveKit agent:

The `LeLamp` class (imported from `lelamp.agent.lelamp_agent`) extends `Agent` from LiveKit:

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
- `LELAMP_EDGE_VISION_ENABLED` (default: false) — Enable local edge vision (face, hand, object)
- `LELAMP_PROACTIVE_MONITOR` (default: true) — Auto-detect presence and gestures without voice trigger
- `LELAMP_MONITOR_ACTIVE_FPS` (default: 5) — Proactive monitor FPS when user present
- `LELAMP_MONITOR_IDLE_FPS` (default: 1) — Proactive monitor FPS when user absent
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
- `LELAMP_PROACTIVE_MONITOR` (default: true) — Proactive vision monitoring
- `LELAMP_MONITOR_ACTIVE_FPS` (default: 5) — Monitor FPS when user present
- `LELAMP_MONITOR_IDLE_FPS` (default: 1) — Monitor FPS when idle
- `LOG_LEVEL` (default: "INFO")

**API & Database Configuration**:
- `LELAMP_API_HOST` (default: "0.0.0.0")
- `LELAMP_API_PORT` (default: 8000)
- `LELAMP_API_WORKERS` (default: 1, set to 4+ for production)
- `LELAMP_DB_URL` (default: "sqlite:///lelamp.db", use PostgreSQL for production)
- `LELAMP_LOG_TO_FILE` (default: false)
- `LELAMP_LOG_DIR` (default: "logs")
- `LELAMP_LOG_JSON` (default: false)

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

### Core Application
- `main.py`: Bootstrap entry point for LiveKit agent
- `lelamp/agent/`: Agent architecture with modular tools
- `lelamp/config.py`: Centralized configuration management with type-safe loading
- `lelamp/service/`: Service architecture (base, motors, RGB, vision, privacy)
- `lelamp/integrations/`: External AI service clients (Baidu, Qwen VL) with unified error handling
- `lelamp/utils/`: Rate limiting, security utilities, OTA updates, shared helpers
- `lelamp/cache/`: TTL caching for LLM responses
- `lelamp/follower/` & `lelamp/leader/`: Motor control configurations
- `lelamp/recordings/`: CSV files with motor animation sequences
- `lelamp/test/`: Hardware testing utilities

### API & Database
- `lelamp/api/`: FastAPI REST API and WebSocket server
  - `app.py`: FastAPI application setup
  - `routes/`: API route handlers (devices, settings, system, websocket)
  - `models/`: Pydantic request/response models
  - `services/`: Business logic (WiFi manager, AP manager, config sync)
- `lelamp/database/`: Database layer with SQLAlchemy ORM
  - `models.py`: Conversation, OperationLog, DeviceState, UserSettings
  - `crud.py`: Database CRUD operations
  - `session.py`: Database session management

### Frontend & Deployment
- `web/`: Vue 3 frontend (standalone HTML/JS/CSS)
- `scripts/`: Build, deployment, and systemd service scripts for commercial deployment
- `docs/`: Technical documentation (CAPTIVE_PORTAL_GUIDE, ARCHITECTURE, SECURITY, API, etc.)
- `VERSION`: Current runtime version string (used for OTA updates)

### Documentation
- `docs/CAPTIVE_PORTAL_GUIDE.md`: Captive Portal setup and usage guide
- `docs/ARCHITECTURE.md`: System architecture details
- `docs/API.md`: REST API documentation
- `docs/SECURITY.md`: Security guidelines
- `docs/plans/`: Implementation plans and design documents

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

### Agent Development

When adding new motor animations:
1. Record movement with `uv run -m lelamp.record --id <lamp_id> --port <port> --name <name>`
2. CSV file saved to `lelamp/recordings/<name>.csv`
3. Call via agent tool: `play_recording("<name>")`
4. Recordings are cached in memory after first load for 10x faster replay

### API Development

When adding new API endpoints:
1. Create route handler in `lelamp/api/routes/`
2. Define Pydantic models in `lelamp/api/models/`
3. Implement business logic in `lelamp/api/services/`
4. Add WebSocket message types if real-time updates needed
5. Test using auto-generated docs at `/docs` (Swagger UI)

When adding database models:
1. Define model in `lelamp/database/models.py`
2. Add CRUD operations in `lelamp/database/crud.py`
3. Create migration or use `Base.metadata.create_all()` for development
4. Support both SQLite (dev) and PostgreSQL (production) via environment variables

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

When setting up Captive Portal:
- Captive Portal is a standalone FastAPI app in `lelamp/api/services/captive_portal.py`
- It runs on port 8080 with its own systemd service (`scripts/lelamp-captive-portal.service`)
- Installation script: `scripts/install_captive_portal.sh`
- For development, run directly: `python -m lelamp.api.services.captive_portal`
- Full documentation: `docs/CAPTIVE_PORTAL_GUIDE.md`
- Implement OTA updates using `OTAManager` class for remote firmware updates
- Build distribution packages with `scripts/build_dist.sh` for releases

When debugging services:
- Check logs with `LOG_LEVEL=DEBUG`
- Service state available via `is_running` and `has_pending_event` properties
- Use `wait_until_idle(timeout)` before service shutdown

When debugging API issues:
- Check API logs: `tail -f logs/lelamp.api.log` (if file logging enabled)
- Test endpoints using Swagger UI at `http://localhost:8000/docs`
- Monitor WebSocket messages using browser dev tools or WebSocket clients
- Check database state: `sqlite3 lelamp.db "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 10"`
- Verify CORS configuration if web client can't connect

When debugging agent issues:
- Enable DEBUG logging for detailed conversation flow
- Check rate limiter stats: `get_all_rate_limiter_stats()`
- Monitor motor health: `motors_service.get_motor_health_summary()`
- Test vision independently: call `vision_answer()` tool directly
- Verify service event queues: check for dropped events in logs

When working with edge vision:
- MediaPipe uses **Tasks API** (`mediapipe.tasks.python.vision`), NOT the deprecated `mediapipe.solutions`
- Models are in `models/` directory — download via scripts or directly from Google Cloud Storage
- All edge components have NoOp fallback: set `_noop = True` when MediaPipe unavailable
- To add a new edge capability: create in `lelamp/edge/`, register in `__init__.py`, expose via `edge_vision_tools.py`
- Edge vision tests: `uv run pytest tests/test_edge_vision.py -v`
- Suppress MediaPipe C++ logs: `export GLOG_minloglevel=2`

## Important Implementation Notes

### Architecture Patterns

**Agent Modularity**: The agent code is split across multiple modules:
- `lelamp/agent/lelamp_agent.py`: Main agent class
- `lelamp/agent/tools/`: Function tools grouped by domain
- `lelamp/agent/states.py`: State management
- This modularity makes it easier to add new capabilities and maintain code

**API Layer Separation**:
- Routes: HTTP endpoint handlers
- Services: Business logic (WiFi, AP manager, etc.)
- Models: Pydantic schemas for validation
- Database: ORM models and CRUD operations

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

**Motor Health Monitoring** (`lelamp/service/motors/health_monitor.py`):
- Real-time monitoring of temperature, voltage, load, and position errors
- Health states: HEALTHY, WARNING, CRITICAL, STALLED
- Background health check thread (default 5min intervals)
- Automatic protective actions (stop playback on stall/critical)
- Health history tracking and statistics
- Access via: `get_motor_health_summary()` and `tune_motor_pid()`

**WiFi Management** (`lelamp/api/services/wifi_manager.py`):
- WiFi network scanning and connection management
- AP (Access Point) mode for onboarding new devices
- Network configuration persistence
- Integration with system network manager (NetworkManager on Linux)

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
- **Database**: Indexed queries on lamp_id and timestamp for fast lookups
- **WebSocket**: Real-time state updates without polling overhead
