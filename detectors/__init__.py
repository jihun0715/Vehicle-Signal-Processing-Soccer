"""Detector wrappers used by the soccer pipeline."""

from .yolo_detector import PersonDetection, YoloPersonDetector

__all__ = [
    "PersonDetection",
    "YoloPersonDetector",
]
