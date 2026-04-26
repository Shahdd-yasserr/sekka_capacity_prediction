"""
config.py
=========
Single source of truth for all shared constants.
Import from here in every other module — never duplicate these values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BUFFER_PATH   = MODEL_DIR / "sensor_buffer.pkl"
META_PATH     = MODEL_DIR / "metro_model_meta.pkl"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"   # version log

# Latest "live" model symlink / path (always points to best version)
LIVE_MODEL_PATH = MODEL_DIR / "metro_capacity_model_live.txt"

# ---------------------------------------------------------------------------
# Station topology
# ---------------------------------------------------------------------------
INTERCHANGE_STATIONS: frozenset[int] = frozenset(
    [119, 120, 122, 208, 209, 211, 215, 313, 319, 320, 324, 332, 335]
)

STATION_LINE_MAP: dict[int, int] = {
    **{sid: 1 for sid in range(100, 200)},
    **{sid: 2 for sid in range(200, 300)},
    **{sid: 3 for sid in range(300, 400)},
}

ALL_STATION_IDS: list[int] = list(range(100, 140)) + list(range(200, 220)) + list(range(300, 340))

# ---------------------------------------------------------------------------
# Capacity thresholds
# ---------------------------------------------------------------------------
MAX_CAPACITY  = 100   # people — train is "full" at this number
SEAT_CAPACITY = 40    # people — seats run out above this

# ---------------------------------------------------------------------------
# Sensor validation rules
# ---------------------------------------------------------------------------
SENSOR_SCHEMA: dict = {
    "station_id":   {"type": int,   "min": 100,  "max": 399},
    "people_count": {"type": (int, float), "min": 0, "max": 300},
    # timestamp validated separately (ISO-8601 string)
}

# How many consecutive bad readings before a station is flagged as offline
SENSOR_DROPOUT_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Buffer / rolling window
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS = 30          # keep last N days in the live buffer
LAG_BUFFER_SIZE     = 16          # readings kept per station for lag features (≥ max lag used)

# ---------------------------------------------------------------------------
# Retraining triggers
# ---------------------------------------------------------------------------
RETRAIN_EVERY_N_ROWS   = 10_000   # volume-based trigger
RETRAIN_EVERY_HOURS    = 6        # time-based trigger
MAE_DRIFT_THRESHOLD    = 8.0      # drift-based trigger — retrain if recent MAE > this

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict = {
    "objective":        "regression",
    "metric":           "mae",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "verbose":          -1,
    "n_jobs":           -1,
    "min_data_in_leaf": 50,
    "max_depth":        -1,
}

CATEGORICAL_FEATURES: list[str] = ["station_id", "line_number"]
TRAIN_RATIO: float = 0.8
NUM_BOOST_ROUND: int = 1000
EARLY_STOPPING_ROUNDS: int = 50

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# API keys — two scopes:
#   SENSOR_API_KEYS : allowed to POST sensor readings (hardware / IoT devices)
#   APP_API_KEYS    : allowed to GET predictions (Flutter app, dashboards)
# In production, load these from environment variables or a secrets manager.
# Add as many keys as you have clients.
SENSOR_API_KEYS: set[str] = {
    "sensor-key-line1-abc123",
    "sensor-key-line2-def456",
    "sensor-key-line3-ghi789",
}

APP_API_KEYS: set[str] = {
    "flutter-app-key-xyz999",
    "dashboard-key-uvw888",
}

# Both scopes combined (for endpoints that accept either)
ALL_API_KEYS: set[str] = SENSOR_API_KEYS | APP_API_KEYS

# Rate limiting (requests per minute per API key)
RATE_LIMIT_SENSOR: int = 120   # sensors post every 7 min → plenty of headroom
RATE_LIMIT_APP:    int = 60    # Flutter polls ~every 7 min
