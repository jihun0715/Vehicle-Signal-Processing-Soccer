"""공유 utility 계층의 public API를 모아 export하는 패키지 파일.

주요 클래스:
- `ProjectionConfig`, `ImageToWorldProjector`: image-space detection을 world 좌표 observation으로 바꾼다.
- `KalmanTrackerConfig`, `WorldKalmanTracker`, `WorldObservation`, `WorldTrack`: world-frame Kalman tracking을 담당한다.
- `TrackingVisualizationWriter`: tracking 결과를 상하분할 debug video로 저장한다.
"""

from .kalman_filter import (
    KalmanTrackerConfig,
    WorldKalmanTracker,
    WorldObservation,
    WorldTrack,
)
from .projection import ImageToWorldProjector, ProjectionConfig
from .tracking_video import TrackingVisualizationWriter

__all__ = [
    "ImageToWorldProjector",
    "KalmanTrackerConfig",
    "ProjectionConfig",
    "TrackingVisualizationWriter",
    "WorldKalmanTracker",
    "WorldObservation",
    "WorldTrack",
]
