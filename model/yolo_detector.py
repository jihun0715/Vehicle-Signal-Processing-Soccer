"""Ultralytics YOLO를 person-only detector로 감싼 모듈.

주요 클래스:
- `PersonDetection`: image-space bbox, confidence, camera/frame metadata를 담는 detection 결과 데이터 클래스.
- `YoloPersonDetector`: config의 model path/device/conf threshold를 사용해 YOLO를 로드하고 person class만 검출한다.

이 모듈은 import 시점이 아니라 detector 생성 시점에 `ultralytics`를 import해서,
YOLO가 없는 환경에서도 다른 유틸 모듈은 import 가능하게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PersonDetection:
    """One image-space person detection."""

    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str = "person"
    camera_id: Optional[int] = None
    frame_index: Optional[int] = None
    timestamp_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": self.bbox_xyxy,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "label": self.label,
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
        }


class YoloPersonDetector:
    """Thin, config-driven wrapper around ``ultralytics.YOLO``.

    The import is intentionally delayed until this class is instantiated, so
    non-YOLO utilities can still be imported on hosts without Ultralytics.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        person_class_id: int = 0,
        verbose: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLO detection. "
                "Build/run the project Docker image or install `ultralytics`."
            ) from exc

        self.model_path = str(model_path)
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.person_class_id = int(person_class_id)
        self.verbose = bool(verbose)
        self.model = YOLO(self.model_path)

    @classmethod
    def from_config(cls) -> "YoloPersonDetector":
        import config as project_config

        return cls(
            model_path=project_config.YOLO_MODEL_PATH,
            device=project_config.YOLO_DEVICE,
            conf_threshold=project_config.YOLO_CONF_THRESHOLD,
            iou_threshold=project_config.YOLO_IOU_THRESHOLD,
            person_class_id=project_config.YOLO_PERSON_CLASS_ID,
            verbose=project_config.YOLO_VERBOSE,
        )

    def detect(
        self,
        image: Any,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[PersonDetection]:
        """Run person-only detection on one image.

        ``image`` may be a numpy HWC image or a torch CHW tensor. The project
        dataloaders pass BGR numpy frames when used by the tracking pipeline;
        Ultralytics handles numpy frames directly.
        """

        image_np = _to_numpy_image(image)
        predict_kwargs = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "classes": [self.person_class_id],
            "verbose": self.verbose,
        }
        if self.device not in {None, ""}:
            predict_kwargs["device"] = self.device

        results = self.model.predict(image_np, **predict_kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(np.int64)

        camera_id = _optional_int(meta.get("camera_id")) if meta else None
        frame_index = _optional_int(meta.get("frame_index")) if meta else None
        timestamp_sec = _optional_float(meta.get("timestamp_sec")) if meta else None

        detections: List[PersonDetection] = []
        for bbox, score, class_id in zip(xyxy, conf, cls):
            if int(class_id) != self.person_class_id:
                continue
            detections.append(
                PersonDetection(
                    bbox_xyxy=tuple(float(value) for value in bbox),
                    confidence=float(score),
                    class_id=int(class_id),
                    camera_id=camera_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                )
            )
        return detections


def _to_numpy_image(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("image must not be None for YOLO detection")

    if hasattr(image, "detach") and hasattr(image, "cpu"):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
