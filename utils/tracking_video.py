"""tracking 결과를 상하분할 debug video로 시각화하는 모듈.

주요 클래스:
- `FieldRenderConfig`: 미니맵 field 크기, 출력 해상도, covariance ellipse sigma를 담는다.
- `TrackingVisualizationWriter`: 카메라별 mp4 writer를 관리하며 원본 영상과 field minimap을 합쳐 저장한다.
- `_WorldToMapTransform`: world 좌표 `(x, y)`를 field minimap pixel 좌표로 바꾸는 내부 helper다.

주요 함수:
- `build_tracking_canvas`: 위에는 카메라 영상, 아래에는 field map을 둔 하나의 frame을 만든다.
- `render_field_map`: 축구장 라인, track center, covariance ellipse를 그린다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FieldRenderConfig:
    field_length_m: float
    field_width_m: float
    output_width: int
    output_height: int
    covariance_sigma: float = 2.0


class TrackingVisualizationWriter:
    """Write one top/bottom tracking visualization video per camera."""

    def __init__(
        self,
        output_dir: Path,
        camera_ids: Sequence[int],
        *,
        fps: float,
        output_width: int,
        field_height: int,
        field_length_m: float,
        field_width_m: float,
        draw_detections: bool = True,
        covariance_sigma: float = 2.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.camera_ids = tuple(int(camera_id) for camera_id in camera_ids)
        self.fps = float(fps)
        self.draw_detections = bool(draw_detections)
        self.field_config = FieldRenderConfig(
            field_length_m=float(field_length_m),
            field_width_m=float(field_width_m),
            output_width=int(output_width),
            output_height=int(field_height),
            covariance_sigma=float(covariance_sigma),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writers: Dict[int, cv2.VideoWriter] = {}
        self.output_paths: Dict[int, Path] = {
            camera_id: self.output_dir / f"tracking_cam{camera_id}.mp4"
            for camera_id in self.camera_ids
        }

    @classmethod
    def from_config(cls) -> "TrackingVisualizationWriter":
        import config as project_config

        return cls(
            output_dir=project_config.PIPELINE_VIS_OUTPUT_DIR,
            camera_ids=project_config.PIPELINE_CAMERAS,
            fps=project_config.PIPELINE_VIS_FPS,
            output_width=project_config.PIPELINE_VIS_OUTPUT_WIDTH,
            field_height=project_config.PIPELINE_VIS_FIELD_HEIGHT,
            field_length_m=project_config.FIELD_LENGTH_M,
            field_width_m=project_config.FIELD_WIDTH_M,
            draw_detections=project_config.PIPELINE_VIS_DRAW_DETECTIONS,
            covariance_sigma=project_config.PIPELINE_VIS_COVARIANCE_SIGMA,
        )

    def write(self, sample: Dict[str, Any], result: Dict[str, Any]) -> None:
        for camera_id in self.camera_ids:
            camera_sample = _get_by_camera_id(sample.get("cameras", {}), camera_id)
            camera_result = _get_by_camera_id(result.get("cameras", {}), camera_id)
            if camera_sample is None or camera_result is None:
                continue

            canvas = build_tracking_canvas(
                camera_sample=camera_sample,
                camera_result=camera_result,
                field_config=self.field_config,
                draw_detections=self.draw_detections,
            )
            writer = self._get_writer(camera_id, canvas.shape[1], canvas.shape[0])
            writer.write(canvas)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

    def _get_writer(self, camera_id: int, width: int, height: int) -> cv2.VideoWriter:
        camera_id = int(camera_id)
        writer = self._writers.get(camera_id)
        if writer is not None:
            return writer

        output_path = self.output_paths[camera_id]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open visualization video writer: {output_path}")
        self._writers[camera_id] = writer
        return writer


def build_tracking_canvas(
    *,
    camera_sample: Dict[str, Any],
    camera_result: Dict[str, Any],
    field_config: FieldRenderConfig,
    draw_detections: bool = True,
) -> np.ndarray:
    top = _to_bgr(camera_sample["image"]).copy()
    if draw_detections:
        _draw_detections(top, camera_result.get("detections", ()))
    _draw_image_overlay(top, camera_sample, camera_result)
    top = _resize_to_width(top, field_config.output_width)

    field = render_field_map(camera_result, field_config)
    if field.shape[1] != top.shape[1]:
        field = cv2.resize(field, (top.shape[1], field.shape[0]), interpolation=cv2.INTER_AREA)
    return np.vstack([top, field])


def render_field_map(camera_result: Dict[str, Any], field_config: FieldRenderConfig) -> np.ndarray:
    width = int(field_config.output_width)
    height = int(field_config.output_height)
    canvas = np.full((height, width, 3), (26, 92, 42), dtype=np.uint8)

    transform = _WorldToMapTransform(field_config, margin=24)
    _draw_pitch(canvas, transform)
    _draw_tracks(canvas, transform, camera_result.get("tracks", ()), field_config.covariance_sigma)
    _draw_field_overlay(canvas, camera_result)
    return canvas


def _draw_pitch(canvas: np.ndarray, transform: "_WorldToMapTransform") -> None:
    white = (235, 245, 235)
    thin = 1
    thick = 2
    length = transform.field_length_m
    width = transform.field_width_m
    half_l = length / 2.0
    half_w = width / 2.0

    cv2.rectangle(
        canvas,
        transform.to_px((-half_l, -half_w)),
        transform.to_px((half_l, half_w)),
        white,
        thick,
    )
    cv2.line(canvas, transform.to_px((0.0, -half_w)), transform.to_px((0.0, half_w)), white, thin)
    cv2.circle(canvas, transform.to_px((0.0, 0.0)), transform.radius_to_px(9.15), white, thin)
    cv2.circle(canvas, transform.to_px((0.0, 0.0)), 3, white, -1)

    penalty_depth = 16.5
    penalty_width = 40.32
    goal_depth = 5.5
    goal_width = 18.32
    penalty_spot_dx = 11.0
    for sign in (-1.0, 1.0):
        goal_x = sign * half_l
        penalty_inner_x = sign * (half_l - penalty_depth)
        goal_inner_x = sign * (half_l - goal_depth)
        spot_x = sign * (half_l - penalty_spot_dx)

        _draw_box(canvas, transform, goal_x, penalty_inner_x, penalty_width, white)
        _draw_box(canvas, transform, goal_x, goal_inner_x, goal_width, white)
        cv2.circle(canvas, transform.to_px((spot_x, 0.0)), 2, white, -1)

    _draw_axis_hint(canvas, transform)


def _draw_box(
    canvas: np.ndarray,
    transform: "_WorldToMapTransform",
    edge_x: float,
    inner_x: float,
    box_width: float,
    color: Tuple[int, int, int],
) -> None:
    y0 = -box_width / 2.0
    y1 = box_width / 2.0
    cv2.rectangle(canvas, transform.to_px((edge_x, y0)), transform.to_px((inner_x, y1)), color, 1)


def _draw_axis_hint(canvas: np.ndarray, transform: "_WorldToMapTransform") -> None:
    origin = transform.to_px((0.0, 0.0))
    east = transform.to_px((12.0, 0.0))
    north = transform.to_px((0.0, 8.0))
    cv2.arrowedLine(canvas, origin, east, (210, 235, 210), 1, tipLength=0.25)
    cv2.arrowedLine(canvas, origin, north, (210, 235, 210), 1, tipLength=0.25)
    cv2.putText(
        canvas,
        "+x E",
        (east[0] + 4, east[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (230, 245, 230),
        1,
    )
    cv2.putText(
        canvas,
        "+y N",
        (north[0] + 4, north[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (230, 245, 230),
        1,
    )


def _draw_tracks(
    canvas: np.ndarray,
    transform: "_WorldToMapTransform",
    tracks: Iterable[Dict[str, Any]],
    covariance_sigma: float,
) -> None:
    for track in tracks:
        state = _as_float_array(track.get("state"))
        if state.size < 2:
            position = _as_float_array(track.get("position"))
        else:
            position = state[:2]
        if position.size < 2 or not np.all(np.isfinite(position[:2])):
            continue

        track_id = int(track.get("track_id", -1))
        color = _track_color(track_id)
        center = transform.to_px((float(position[0]), float(position[1])))

        covariance = _position_covariance(track.get("covariance"))
        if covariance is not None:
            _draw_covariance_ellipse(canvas, transform, center, covariance, color, covariance_sigma)

        cv2.circle(canvas, center, 4, color, -1)
        cv2.circle(canvas, center, 6, (10, 20, 10), 1)

        label = f"id {track_id}"
        cv2.putText(
            canvas,
            label,
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            label,
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


def _draw_covariance_ellipse(
    canvas: np.ndarray,
    transform: "_WorldToMapTransform",
    center: Tuple[int, int],
    covariance: np.ndarray,
    color: Tuple[int, int, int],
    sigma: float,
) -> None:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return

    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    axes_m = max(float(sigma), 0.0) * np.sqrt(eigvals)
    axes_px = np.maximum(np.round(axes_m * transform.scale).astype(int), 1)
    axes_px = np.minimum(axes_px, max(canvas.shape[0], canvas.shape[1]))

    major_vec = eigvecs[:, 0]
    pixel_vec = np.array([major_vec[0], -major_vec[1]], dtype=np.float64)
    angle_deg = float(np.degrees(np.arctan2(pixel_vec[1], pixel_vec[0])))

    overlay = canvas.copy()
    cv2.ellipse(
        overlay,
        center,
        (int(axes_px[0]), int(axes_px[1])),
        angle_deg,
        0.0,
        360.0,
        color,
        -1,
    )
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, canvas)
    cv2.ellipse(
        canvas,
        center,
        (int(axes_px[0]), int(axes_px[1])),
        angle_deg,
        0.0,
        360.0,
        color,
        2,
    )


def _draw_field_overlay(canvas: np.ndarray, camera_result: Dict[str, Any]) -> None:
    text_lines = [
        f"cam {camera_result.get('camera_id')}",
        f"frame {camera_result.get('frame_index')}",
        f"tracks {len(camera_result.get('tracks', ()))}, confirmed {len(camera_result.get('confirmed_tracks', ()))}",
        "world: center origin, +x east, +y north",
    ]
    _draw_text_block(canvas, text_lines, (18, 30), scale=0.55, line_height=22)


def _draw_detections(frame: np.ndarray, detections: Iterable[Dict[str, Any]]) -> None:
    for detection in detections:
        bbox = detection.get("bbox_xyxy")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        confidence = float(detection.get("confidence", 0.0))
        color = (70, 230, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"person {confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def _draw_image_overlay(
    frame: np.ndarray,
    camera_sample: Dict[str, Any],
    camera_result: Dict[str, Any],
) -> None:
    meta = camera_sample.get("meta", {})
    text_lines = [
        f"cam {camera_result.get('camera_id', meta.get('camera_id'))}",
        f"frame {camera_result.get('frame_index', meta.get('frame_index'))}",
        f"det {len(camera_result.get('detections', ()))}, tracks {len(camera_result.get('tracks', ()))}",
    ]
    _draw_text_block(frame, text_lines, (18, 30), scale=0.65, line_height=25)


def _draw_text_block(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int],
    *,
    scale: float,
    line_height: int,
) -> None:
    x, y = origin
    box_width = max(220, int(max((len(line) for line in lines), default=0) * 11 * scale))
    box_height = line_height * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 22), (x + box_width, y - 22 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            2,
        )


def _resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    height, current_width = frame.shape[:2]
    if current_width == width:
        return frame
    scale = width / float(current_width)
    return cv2.resize(frame, (int(width), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def _to_bgr(image: Any) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim == 3 and frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if np.issubdtype(frame.dtype, np.floating):
        if frame.size and float(np.nanmax(frame)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _get_by_camera_id(mapping: Dict[Any, Any], camera_id: int) -> Optional[Any]:
    if camera_id in mapping:
        return mapping[camera_id]
    return mapping.get(str(camera_id))


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _position_covariance(value: Any) -> Optional[np.ndarray]:
    covariance = _as_float_array(value)
    if covariance.ndim >= 2 and covariance.shape[0] >= 2 and covariance.shape[1] >= 2:
        return covariance[:2, :2]
    if covariance.size >= 4:
        return covariance.reshape(-1, 2)[:2, :2]
    return None


def _track_color(track_id: int) -> Tuple[int, int, int]:
    palette: List[Tuple[int, int, int]] = [
        (0, 220, 255),
        (255, 180, 0),
        (80, 255, 90),
        (255, 90, 180),
        (180, 120, 255),
        (255, 255, 90),
        (90, 180, 255),
        (120, 255, 220),
    ]
    if track_id < 0:
        return (220, 220, 220)
    return palette[track_id % len(palette)]


class _WorldToMapTransform:
    def __init__(self, config: FieldRenderConfig, margin: int) -> None:
        self.field_length_m = float(config.field_length_m)
        self.field_width_m = float(config.field_width_m)
        self.margin = int(margin)
        self.canvas_width = int(config.output_width)
        self.canvas_height = int(config.output_height)
        usable_width = max(1, self.canvas_width - 2 * self.margin)
        usable_height = max(1, self.canvas_height - 2 * self.margin)
        self.scale = min(
            usable_width / max(self.field_length_m, 1e-9),
            usable_height / max(self.field_width_m, 1e-9),
        )
        self.field_px_width = self.field_length_m * self.scale
        self.field_px_height = self.field_width_m * self.scale
        self.origin_x = self.canvas_width / 2.0
        self.origin_y = self.canvas_height / 2.0

    def to_px(self, point_xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = point_xy
        px = self.origin_x + float(x) * self.scale
        py = self.origin_y - float(y) * self.scale
        return (int(round(px)), int(round(py)))

    def radius_to_px(self, radius_m: float) -> int:
        return max(1, int(round(float(radius_m) * self.scale)))
