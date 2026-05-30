"""Project-level configuration.

Edit this file when local paths or default debug settings change.
Environment variables with the same purpose can still override these defaults.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# In Docker, docker.sh overrides this with /datasets/ISSIA-Soccer.
# For another host path, edit this default or pass --root to debug.py.
DEFAULT_ISSIA_SOCCER_ROOT = Path("/media/jihun/Crucial X10/ISSIA-Soccer")
ISSIA_SOCCER_ROOT = Path(os.environ.get("ISSIA_SOCCER_ROOT", DEFAULT_ISSIA_SOCCER_ROOT))

ISSIA_CAMERAS = (1, 2, 3, 4, 5, 6)
ISSIA_BALL_BBOX_SIZE = 20

DEBUG_OUTPUT_DIR = PROJECT_ROOT / "debug_outputs"
