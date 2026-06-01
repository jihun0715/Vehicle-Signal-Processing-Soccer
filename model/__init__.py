"""모델 실행 계층의 public API를 모아 export하는 패키지 파일.

주요 클래스:
- `PersonDetection`, `YoloPersonDetector`: YOLO 기반 person detection 결과와 wrapper.
- `SoccerTrackingPipeline`: YOLO -> projection -> Kalman tracking을 연결하는 end-to-end pipeline.

외부 코드는 되도록 이 패키지에서 필요한 모델 클래스를 import하도록 한다.
"""

from .soccer_tracking import SoccerTrackingPipeline, summarize_camera_result
from .yolo_detector import PersonDetection, YoloPersonDetector

__all__ = [
    "PersonDetection",
    "SoccerTrackingPipeline",
    "YoloPersonDetector",
    "summarize_camera_result",
]
