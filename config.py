"""Project-level configuration.

Edit this file when local paths or default debug settings change.
Environment variables with the same purpose can still override these defaults.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# In Docker, docker.sh overrides this with /ssd/ISSIA-Soccer.
# For another host path, edit this default or pass --root to debug.py.
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
