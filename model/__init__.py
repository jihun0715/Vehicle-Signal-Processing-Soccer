"""Model-facing entry points for detection and tracking pipelines."""

from .soccer_tracking import SoccerTrackingPipeline, summarize_camera_result
from .yolo_detector import PersonDetection, YoloPersonDetector

__all__ = [
    "PersonDetection",
    "SoccerTrackingPipeline",
    "YoloPersonDetector",
    "summarize_camera_result",
]
