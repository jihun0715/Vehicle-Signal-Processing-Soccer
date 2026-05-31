"""Projection helpers from image-space detections to a common world frame."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tracking import WorldObservation


@dataclass(frozen=True)
class ProjectionConfig:
    homographies: Mapping[Any, Any] = field(default_factory=dict)
    homography_path: Optional[Any] = None
    image_reference_points: Mapping[Any, Any] = field(default_factory=dict)
    world_reference_points: Mapping[Any, Any] = field(default_factory=dict)
    fallback_mode: str = "normalized_image"
    field_length_m: float = 105.0
    field_width_m: float = 68.0

    @classmethod
    def from_config(cls) -> "ProjectionConfig":
        import config as project_config

        return cls(
            homographies=project_config.CAMERA_IMAGE_TO_WORLD_HOMOGRAPHIES,
            homography_path=project_config.CAMERA_IMAGE_TO_WORLD_HOMOGRAPHIES_PATH,
            image_reference_points=project_config.CAMERA_IMAGE_REFERENCE_POINTS,
            world_reference_points=project_config.WORLD_REFERENCE_POINTS,
            fallback_mode=project_config.PROJECTION_FALLBACK_MODE,
            field_length_m=project_config.FIELD_LENGTH_M,
            field_width_m=project_config.FIELD_WIDTH_M,
        )


class ImageToWorldProjector:
    """Project player foot points into a shared pitch/world coordinate frame."""

    def __init__(self, projection_config: Optional[ProjectionConfig] = None) -> None:
        self.config = projection_config or ProjectionConfig()
        self._homographies = _load_homographies_from_path(self.config.homography_path)
        self._homographies.update(_normalize_homographies(self.config.homographies))
        self._homographies.update(
            _homographies_from_reference_points(
                self.config.image_reference_points,
                self.config.world_reference_points,
            )
        )
        if self.config.fallback_mode not in {"normalized_image", "raise"}:
            raise ValueError("fallback_mode must be 'normalized_image' or 'raise'")

    @classmethod
    def from_config(cls) -> "ImageToWorldProjector":
        return cls(ProjectionConfig.from_config())

    def project_bbox_to_world(
        self,
        bbox_xyxy: Sequence[float],
        *,
        camera_id: int,
        image_size: Tuple[int, int],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Project the bbox bottom-center point into world coordinates."""

        foot_point = bbox_bottom_center(bbox_xyxy)
        world_x, world_y, metadata = self.project_point_to_world(
            foot_point,
            camera_id=camera_id,
            image_size=image_size,
        )
        metadata["image_point_xy"] = foot_point
        return world_x, world_y, metadata

    def project_point_to_world(
        self,
        point_xy: Tuple[float, float],
        *,
        camera_id: int,
        image_size: Tuple[int, int],
    ) -> Tuple[float, float, Dict[str, Any]]:
        homography = self._homographies.get(int(camera_id))
        if homography is not None:
            projected = homography.dot(np.array([point_xy[0], point_xy[1], 1.0], dtype=np.float64))
            scale = float(projected[2])
            if abs(scale) < 1e-9:
                raise ValueError(f"Invalid homography projection scale for camera {camera_id}")
            return (
                float(projected[0] / scale),
                float(projected[1] / scale),
                {"projection_mode": "homography", "camera_id": int(camera_id)},
            )

        if self.config.fallback_mode == "raise":
            raise KeyError(f"Missing image-to-world homography for camera {camera_id}")

        height, width = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image_size: {image_size}")
        world_x = (
            float(point_xy[0]) / float(width) * float(self.config.field_length_m)
            - float(self.config.field_length_m) * 0.5
        )
        world_y = (
            float(self.config.field_width_m) * 0.5
            - float(point_xy[1]) / float(height) * float(self.config.field_width_m)
        )
        return (
            world_x,
            world_y,
            {
                "projection_mode": "normalized_image",
                "camera_id": int(camera_id),
                "warning": "No homography configured; using normalized image fallback.",
            },
        )

    def detections_to_observations(
        self,
        detections: Iterable[Any],
        *,
        camera_id: int,
        image_size: Tuple[int, int],
        timestamp_sec: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> List[WorldObservation]:
        observations: List[WorldObservation] = []
        for detection_index, detection in enumerate(detections):
            bbox = _get_bbox_xyxy(detection)
            world_x, world_y, projection_meta = self.project_bbox_to_world(
                bbox,
                camera_id=camera_id,
                image_size=image_size,
            )
            confidence = _get_confidence(detection)
            det_camera_id = _get_optional_int(detection, "camera_id")
            det_frame_index = _get_optional_int(detection, "frame_index")
            det_timestamp = _get_optional_float(detection, "timestamp_sec")
            metadata = {
                "projection": projection_meta,
                "detection": _detection_to_dict(detection),
            }
            observations.append(
                WorldObservation(
                    x=world_x,
                    y=world_y,
                    timestamp_sec=det_timestamp if det_timestamp is not None else timestamp_sec,
                    confidence=confidence,
                    camera_id=det_camera_id if det_camera_id is not None else int(camera_id),
                    frame_index=det_frame_index if det_frame_index is not None else frame_index,
                    bbox_xyxy=tuple(float(value) for value in bbox),
                    detection_id=detection_index,
                    metadata=metadata,
                )
            )
        return observations


def bbox_bottom_center(bbox_xyxy: Sequence[float]) -> Tuple[float, float]:
    x1, _y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return ((x1 + x2) * 0.5, y2)


def _normalize_homographies(raw_homographies: Mapping[Any, Any]) -> Dict[int, np.ndarray]:
    homographies: Dict[int, np.ndarray] = {}
    for raw_key, raw_value in dict(raw_homographies).items():
        camera_id = int(raw_key)
        matrix = np.asarray(raw_value, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Camera {camera_id} homography must be 3x3, got {matrix.shape}")
        homographies[camera_id] = matrix
    return homographies


def _load_homographies_from_path(path: Optional[Any]) -> Dict[int, np.ndarray]:
    if path in {None, ""}:
        return {}

    homography_path = Path(path)
    if not homography_path.exists():
        return {}

    with homography_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    raw_homographies = payload.get("homographies", payload)
    return _normalize_homographies(raw_homographies)


def _homographies_from_reference_points(
    raw_image_points: Mapping[Any, Any],
    raw_world_points: Mapping[Any, Any],
) -> Dict[int, np.ndarray]:
    if not raw_image_points or not raw_world_points:
        return {}

    world_points = {
        str(label): tuple(float(value) for value in xy)
        for label, xy in dict(raw_world_points).items()
    }
    homographies = {}
    for raw_camera_id, camera_points_raw in dict(raw_image_points).items():
        camera_id = int(raw_camera_id)
        camera_points = {
            str(label): tuple(float(value) for value in xy)
            for label, xy in dict(camera_points_raw).items()
        }
        shared_labels = sorted(set(camera_points) & set(world_points))
        if len(shared_labels) < 4:
            raise ValueError(
                f"Camera {camera_id} needs at least 4 shared calibration labels, "
                f"got {len(shared_labels)}"
            )
        src = np.asarray([camera_points[label] for label in shared_labels], dtype=np.float64)
        dst = np.asarray([world_points[label] for label in shared_labels], dtype=np.float64)
        homography, _mask = _find_homography(src, dst)
        homographies[camera_id] = homography
    return homographies


def _find_homography(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[np.ndarray, Any]:
    try:
        import cv2

        homography, mask = cv2.findHomography(src_points, dst_points, method=0)
        if homography is None:
            raise ValueError("cv2.findHomography returned None")
        return homography.astype(np.float64), mask
    except ImportError:
        return _direct_linear_transform_homography(src_points, dst_points), None


def _direct_linear_transform_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    rows = []
    for (src_x, src_y), (dst_x, dst_y) in zip(src_points, dst_points):
        rows.append([-src_x, -src_y, -1.0, 0.0, 0.0, 0.0, dst_x * src_x, dst_x * src_y, dst_x])
        rows.append([0.0, 0.0, 0.0, -src_x, -src_y, -1.0, dst_y * src_x, dst_y * src_y, dst_y])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homography = vh[-1].reshape(3, 3)
    scale = homography[2, 2]
    if abs(float(scale)) > 1e-12:
        homography = homography / scale
    return homography


def _get_bbox_xyxy(detection: Any) -> Tuple[float, float, float, float]:
    if isinstance(detection, dict):
        bbox = detection["bbox_xyxy"]
    else:
        bbox = getattr(detection, "bbox_xyxy")
    return tuple(float(value) for value in bbox)


def _get_confidence(detection: Any) -> float:
    if isinstance(detection, dict):
        return float(detection.get("confidence", 1.0))
    return float(getattr(detection, "confidence", 1.0))


def _get_optional_int(detection: Any, name: str) -> Optional[int]:
    value = _get_optional_value(detection, name)
    return int(value) if value is not None else None


def _get_optional_float(detection: Any, name: str) -> Optional[float]:
    value = _get_optional_value(detection, name)
    return float(value) if value is not None else None


def _get_optional_value(detection: Any, name: str) -> Any:
    if isinstance(detection, dict):
        return detection.get(name)
    return getattr(detection, name, None)


def _detection_to_dict(detection: Any) -> Dict[str, Any]:
    if hasattr(detection, "to_dict"):
        return detection.to_dict()
    if isinstance(detection, dict):
        return dict(detection)
    return {"bbox_xyxy": _get_bbox_xyxy(detection), "confidence": _get_confidence(detection)}
