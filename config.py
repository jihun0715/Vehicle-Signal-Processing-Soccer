"""Project-level configuration.

Edit this file when local paths or default debug settings change.
Environment variables with the same purpose can still override these defaults.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# In Docker, docker.sh overrides this with /ssd/ISSIA-Soccer.
# For another host path, edit this default or set ISSIA_SOCCER_ROOT.
DEFAULT_ISSIA_SOCCER_ROOT = Path("/media/jihun/Crucial X10/ISSIA-Soccer")
ISSIA_SOCCER_ROOT = Path(os.environ.get("ISSIA_SOCCER_ROOT", DEFAULT_ISSIA_SOCCER_ROOT))

ISSIA_CAMERAS = (1, 2, 3, 4, 5, 6)
ISSIA_BALL_BBOX_SIZE = 20

DEBUG_OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"

# Dataloader defaults
DATALOADER_BATCH_SIZE = 4
DATALOADER_NUM_WORKERS = 8
DATALOADER_PIN_MEMORY = False

# General debug defaults
DEBUG_CAMERAS = (1, 2)
DEBUG_FRAME_STEP = 500
DEBUG_SAVE_PREVIEW = False

# Offset-sequence defaults
OFFSET_RANDOM_SEED = 7
OFFSET_MAX_FRAME_OFFSET = 80
OFFSET_REFERENCE_CAMERA = DEBUG_CAMERAS[0]
OFFSET_ALLOW_ZERO_NON_REFERENCE = False
OFFSET_FORCE_REFERENCE_ZERO = True

# Debug video defaults
DEBUG_VIDEO_CAMERAS = DEBUG_CAMERAS
DEBUG_VIDEO_BATCH_SIZE = 1
DEBUG_VIDEO_NUM_WORKERS = 0
DEBUG_VIDEO_START_FRAME = 500
DEBUG_VIDEO_NUM_FRAMES = 1000
DEBUG_VIDEO_FRAME_STEP = 1
DEBUG_VIDEO_WIDTH = 960
DEBUG_VIDEO_FPS = 25.0
DEBUG_VIDEO_SHOW_BOXES = False

# YOLO person detector defaults
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", str(PROJECT_ROOT / "weights" / "yolo26x.pt"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "0")
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.70
YOLO_PERSON_CLASS_ID = 0
YOLO_VERBOSE = False

# Projection defaults
# Fill this with camera_id -> 3x3 image-to-world homography when calibration is ready.
CAMERA_IMAGE_TO_WORLD_HOMOGRAPHIES = {}
CAMERA_IMAGE_TO_WORLD_HOMOGRAPHIES_PATH = Path(
    os.environ.get(
        "CAMERA_IMAGE_TO_WORLD_HOMOGRAPHIES_PATH",
        DEBUG_OUTPUT_DIR / "calibration" / "homographies.json",
    )
)
# Optional calibration-point route. The ISSIA Annotation directory contains
# Reference-Camera-{id}.bmp images with visible Fisso/Giallo/Bianco labels.
# Put clicked image points here as camera_id -> label -> (x, y), and put the
# corresponding field/world coordinates in WORLD_REFERENCE_POINTS.
CAMERA_IMAGE_REFERENCE_POINTS = {}
WORLD_REFERENCE_POINTS = {}
ISSIA_REFERENCE_IMAGE_PATTERN = "Reference-Camera-{camera_id}.bmp"
# Use "raise" after real calibration is configured. The fallback is only for
# early pipeline smoke tests before camera calibration files exist.
PROJECTION_FALLBACK_MODE = "normalized_image"  # "normalized_image" or "raise"
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0

# INHA-style world-frame Kalman tracker defaults
TRACK_GATE_CHI2 = 9.21
TRACK_BIG_COST = 1e9
TRACK_COST_MAHALANOBIS_WEIGHT = 1.0
TRACK_COST_L2_WEIGHT = 0.0
TRACK_COST_L2_NORM_M = 1.0
TRACK_Q_POS = 0.02
TRACK_Q_VEL = 0.50
TRACK_BRAKE_ACCEL = 0.8
TRACK_MIN_SPEED_EPS = 0.03
TRACK_MAX_PREDICT_DT_SEC = 0.20
TRACK_INIT_POS_STD = 2.0
TRACK_INIT_VEL_STD = 1.0
TRACK_MEASUREMENT_STD = 0.25
TRACK_RELIABILITY_SIGMA_REF = 0.60
TRACK_MAX_MISSES = 30
TRACK_MIN_HITS = 1

# End-to-end tracking debug defaults
PIPELINE_CAMERAS = DEBUG_CAMERAS
PIPELINE_START_FRAME = 500
PIPELINE_NUM_FRAMES = 100
PIPELINE_FRAME_STEP = 1
PIPELINE_BATCH_SIZE = 1
PIPELINE_NUM_WORKERS = 0
PIPELINE_OUTPUT_PATH = DEBUG_OUTPUT_DIR / "tracking_results.jsonl"

# Interactive calibration defaults
CALIBRATION_CAMERAS = DEBUG_CAMERAS
CALIBRATION_OUTPUT_DIR = DEBUG_OUTPUT_DIR / "calibration"
CALIBRATION_WINDOW_MAX_WIDTH = 1600
CALIBRATION_WINDOW_MAX_HEIGHT = 900
