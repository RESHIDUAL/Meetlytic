"""
Global configuration settings and signal processing parameters.
All thresholds, sampling rates, and buffer dimensions are defined here.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

SAMPLE_RATE: int = 16000
FRAME_SIZE_MS: int = 25
HOP_SIZE_MS: int = 10
FRAME_SIZE_SAMPLES: int = int(SAMPLE_RATE * FRAME_SIZE_MS / 1000)
HOP_SIZE_SAMPLES: int = int(SAMPLE_RATE * HOP_SIZE_MS / 1000)
AUDIO_BLOCK_SIZE: int = 1600
AUDIO_DTYPE: str = 'int16'

BUFFER_DURATION_SEC: int = 30
BUFFER_SIZE_SAMPLES: int = SAMPLE_RATE * BUFFER_DURATION_SEC
UNCERTAIN_AUTO_DELETE_SEC: int = 60

VAD_NOISE_CALIBRATION_MS: int = 300
VAD_ONSET_FRAMES: int = 3
VAD_HANGOVER_FRAMES: int = 12
VAD_LOOKBACK_FRAMES: int = 2
VAD_ENERGY_MULTIPLIER: float = 3.0
VAD_SFM_THRESHOLD: float = 0.42

SEGMENT_MIN_DURATION_MS: int = 250
SEGMENT_MAX_DURATION_MS: int = 10000
SEGMENT_MERGE_GAP_MS: int = 350

NFFT: int = 512
NUM_MEL_FILTERS: int = 26
NUM_MFCC: int = 13
PRE_EMPHASIS_COEFF: float = 0.97
MEL_LOW_FREQ_HZ: float = 0.0
MEL_HIGH_FREQ_HZ: float = 8000.0

DTW_SAKOE_CHIBA_BAND: int = 15
DTW_DISTANCE_THRESHOLD: float = 1.15

CLASSIFICATION_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    'personal': (0.00, 0.30),
    'casual': (0.30, 0.50),
    'uncertain': (0.50, 0.70),
    'professional': (0.70, 1.00),
}

STRICT_PROFESSIONAL_THRESHOLD: float = 0.85
BALANCED_PROFESSIONAL_THRESHOLD: float = 0.70

DEFAULT_DB_PATH: str = "meeting_records.db"
TEMPLATES_DIR: str = "speech/templates"
MODEL_SAVE_PATH: str = "config/trained_model.json"
WIFI_BRIDGE_PORT: int = 5555
WIFI_BRIDGE_HOST: str = "0.0.0.0"
