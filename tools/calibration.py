"""Interactive ISSIA image-to-world homography calibration tool.

The tool opens ISSIA Reference-Camera-{id}.bmp images, lets you click standard
soccer-pitch landmarks, and saves image-to-world homographies for the tracking
pipeline.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

import config
from data import find_issia_reference_image


FIELD_LENGTH_M = float(config.FIELD_LENGTH_M)
FIELD_WIDTH_M = float(config.FIELD_WIDTH_M)
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
PENALTY_SPOT_DISTANCE_M = 11.0
CENTER_CIRCLE_RADIUS_M = 9.15


def _west_x(distance_from_west_goal_line_m: float = 0.0) -> float:
    return -FIELD_LENGTH_M * 0.5 + float(distance_from_west_goal_line_m)


def _east_x(distance_from_east_goal_line_m: float = 0.0) -> float:
    return FIELD_LENGTH_M * 0.5 - float(distance_from_east_goal_line_m)


def _north_y(width_m: float) -> float:
    return float(width_m) * 0.5


def _south_y(width_m: float) -> float:
    return -float(width_m) * 0.5


# World coordinate convention:
# - origin: center of the field
# - +x: East, along the long axis
# - +y: North, along the short axis
LANDMARKS = {
    "NW_CORNER": (_west_x(), _north_y(FIELD_WIDTH_M)),
    "NE_CORNER": (_east_x(), _north_y(FIELD_WIDTH_M)),
    "SE_CORNER": (_east_x(), _south_y(FIELD_WIDTH_M)),
    "SW_CORNER": (_west_x(), _south_y(FIELD_WIDTH_M)),
    "CENTER": (0.0, 0.0),
    "HALF_NORTH": (0.0, _north_y(FIELD_WIDTH_M)),
    "HALF_SOUTH": (0.0, _south_y(FIELD_WIDTH_M)),
    "WEST_GOAL_CENTER": (_west_x(), 0.0),
    "EAST_GOAL_CENTER": (_east_x(), 0.0),
    "WEST_PENALTY_SPOT": (_west_x(PENALTY_SPOT_DISTANCE_M), 0.0),
    "EAST_PENALTY_SPOT": (_east_x(PENALTY_SPOT_DISTANCE_M), 0.0),
    "WEST_PENALTY_NW": (_west_x(), _north_y(PENALTY_AREA_WIDTH_M)),
    "WEST_PENALTY_NE": (_west_x(PENALTY_AREA_DEPTH_M), _north_y(PENALTY_AREA_WIDTH_M)),
    "WEST_PENALTY_SE": (_west_x(PENALTY_AREA_DEPTH_M), _south_y(PENALTY_AREA_WIDTH_M)),
    "WEST_PENALTY_SW": (_west_x(), _south_y(PENALTY_AREA_WIDTH_M)),
    "EAST_PENALTY_NW": (_east_x(PENALTY_AREA_DEPTH_M), _north_y(PENALTY_AREA_WIDTH_M)),
    "EAST_PENALTY_NE": (_east_x(), _north_y(PENALTY_AREA_WIDTH_M)),
    "EAST_PENALTY_SE": (_east_x(), _south_y(PENALTY_AREA_WIDTH_M)),
    "EAST_PENALTY_SW": (_east_x(PENALTY_AREA_DEPTH_M), _south_y(PENALTY_AREA_WIDTH_M)),
    "WEST_GOAL_AREA_NW": (_west_x(), _north_y(GOAL_AREA_WIDTH_M)),
    "WEST_GOAL_AREA_NE": (_west_x(GOAL_AREA_DEPTH_M), _north_y(GOAL_AREA_WIDTH_M)),
    "WEST_GOAL_AREA_SE": (_west_x(GOAL_AREA_DEPTH_M), _south_y(GOAL_AREA_WIDTH_M)),
    "WEST_GOAL_AREA_SW": (_west_x(), _south_y(GOAL_AREA_WIDTH_M)),
    "EAST_GOAL_AREA_NW": (_east_x(GOAL_AREA_DEPTH_M), _north_y(GOAL_AREA_WIDTH_M)),
    "EAST_GOAL_AREA_NE": (_east_x(), _north_y(GOAL_AREA_WIDTH_M)),
    "EAST_GOAL_AREA_SE": (_east_x(), _south_y(GOAL_AREA_WIDTH_M)),
    "EAST_GOAL_AREA_SW": (_east_x(GOAL_AREA_DEPTH_M), _south_y(GOAL_AREA_WIDTH_M)),
    "CENTER_CIRCLE_NORTH": (0.0, CENTER_CIRCLE_RADIUS_M),
    "CENTER_CIRCLE_SOUTH": (0.0, -CENTER_CIRCLE_RADIUS_M),
    "CENTER_CIRCLE_WEST": (-CENTER_CIRCLE_RADIUS_M, 0.0),
    "CENTER_CIRCLE_EAST": (CENTER_CIRCLE_RADIUS_M, 0.0),
}

DEFAULT_LANDMARK_ORDER = (
    "NW_CORNER",
    "NE_CORNER",
    "SE_CORNER",
    "SW_CORNER",
    "CENTER",
    "HALF_NORTH",
    "HALF_SOUTH",
    "WEST_PENALTY_NW",
    "WEST_PENALTY_NE",
    "WEST_PENALTY_SE",
    "WEST_PENALTY_SW",
    "EAST_PENALTY_NW",
    "EAST_PENALTY_NE",
    "EAST_PENALTY_SE",
    "EAST_PENALTY_SW",
    "WEST_GOAL_AREA_NW",
    "WEST_GOAL_AREA_NE",
    "WEST_GOAL_AREA_SE",
    "WEST_GOAL_AREA_SW",
    "EAST_GOAL_AREA_NW",
    "EAST_GOAL_AREA_NE",
    "EAST_GOAL_AREA_SE",
    "EAST_GOAL_AREA_SW",
    "WEST_PENALTY_SPOT",
    "EAST_PENALTY_SPOT",
    "WEST_GOAL_CENTER",
    "EAST_GOAL_CENTER",
    "CENTER_CIRCLE_NORTH",
    "CENTER_CIRCLE_SOUTH",
    "CENTER_CIRCLE_WEST",
    "CENTER_CIRCLE_EAST",
)


@dataclass
class CalibrationState:
    camera_id: int
    image_path: Path
    image_bgr: np.ndarray
    output_dir: Path
    scale: float
    window_name: str
    landmark_order: Sequence[str] = DEFAULT_LANDMARK_ORDER
    image_points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    selected_index: int = 0
    last_clicked_label: Optional[str] = None
    status: str = ""

    @property
    def selected_label(self) -> str:
        return self.landmark_order[self.selected_index]

    def select_next(self) -> None:
        self.selected_index = (self.selected_index + 1) % len(self.landmark_order)

    def select_previous(self) -> None:
        self.selected_index = (self.selected_index - 1) % len(self.landmark_order)

    def select_next_missing(self) -> None:
        start = self.selected_index
        for step in range(1, len(self.landmark_order) + 1):
            candidate = (start + step) % len(self.landmark_order)
            if self.landmark_order[candidate] not in self.image_points:
                self.selected_index = candidate
                return
        self.select_next()


def main() -> int:
    print("ISSIA soccer-field calibration")
    print(f"  root={config.ISSIA_SOCCER_ROOT}")
    print(f"  cameras={config.CALIBRATION_CAMERAS}")
    print(f"  output_dir={config.CALIBRATION_OUTPUT_DIR}")
    print("Controls: left-click=set point, n/space=next, p=prev, d=delete, u=undo, r=reset, s=save, q=skip, esc=quit")

    for camera_id in config.CALIBRATION_CAMERAS:
        action = calibrate_camera(int(camera_id))
        if action == "quit":
            break
    return 0


def calibrate_camera(camera_id: int) -> str:
    image_path = find_issia_reference_image(
        config.ISSIA_SOCCER_ROOT,
        camera_id,
        pattern=config.ISSIA_REFERENCE_IMAGE_PATTERN,
    )
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read reference image: {image_path}")
    image_bgr = _to_bgr(image)
    scale = _display_scale(image_bgr.shape, config.CALIBRATION_WINDOW_MAX_WIDTH, config.CALIBRATION_WINDOW_MAX_HEIGHT)
    window_name = f"ISSIA calibration camera {camera_id}"
    state = CalibrationState(
        camera_id=camera_id,
        image_path=image_path,
        image_bgr=image_bgr,
        output_dir=config.CALIBRATION_OUTPUT_DIR,
        scale=scale,
        window_name=window_name,
    )
    state.image_points.update(_load_existing_image_points(config.CALIBRATION_OUTPUT_DIR, camera_id))
    state.status = f"Loaded {len(state.image_points)} existing points"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, _on_mouse, state)
    except cv2.error as exc:
        print(f"OpenCV GUI failed to open: {exc}")
        print("Check Docker DISPLAY/X11 settings before running calibration.py.")
        return "quit"

    while True:
        canvas = _build_canvas(state, for_display=True)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 255:
            continue
        if key == 27:
            cv2.destroyWindow(window_name)
            return "quit"
        if key in {ord("q")}:
            cv2.destroyWindow(window_name)
            return "skip"
        if key in {ord("n"), ord(" ")}:
            state.select_next()
            state.status = f"Selected {state.selected_label}"
        elif key == ord("p"):
            state.select_previous()
            state.status = f"Selected {state.selected_label}"
        elif key == ord("d"):
            removed = state.image_points.pop(state.selected_label, None)
            state.status = f"Deleted {state.selected_label}" if removed else f"No point for {state.selected_label}"
        elif key == ord("u"):
            if state.last_clicked_label is not None:
                state.image_points.pop(state.last_clicked_label, None)
                state.status = f"Undid {state.last_clicked_label}"
                state.last_clicked_label = None
            else:
                state.status = "Nothing to undo"
        elif key == ord("r"):
            state.image_points.clear()
            state.last_clicked_label = None
            state.status = "Reset all points"
        elif key == ord("s"):
            if len(state.image_points) < 4:
                state.status = "Need at least 4 points before saving"
                continue
            _save_calibration(state)
            cv2.destroyWindow(window_name)
            return "saved"


def _on_mouse(event: int, x: int, y: int, _flags: int, state: CalibrationState) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    image_x = float(x) / state.scale
    image_y = float(y) / state.scale
    label = state.selected_label
    state.image_points[label] = (image_x, image_y)
    state.last_clicked_label = label
    world_x, world_y = LANDMARKS[label]
    state.status = (
        f"{label}: image=({image_x:.1f}, {image_y:.1f}) "
        f"world=({world_x:.2f}, {world_y:.2f})"
    )
    state.select_next_missing()


def _save_calibration(state: CalibrationState) -> None:
    labels = [label for label in state.landmark_order if label in state.image_points]
    homography, reprojection_errors, rmse = _compute_homography(labels, state.image_points)

    state.output_dir.mkdir(parents=True, exist_ok=True)
    camera_payload = {
        "camera_id": state.camera_id,
        "image_path": str(state.image_path),
        "field_length_m": FIELD_LENGTH_M,
        "field_width_m": FIELD_WIDTH_M,
        "world_coordinate_system": {
            "origin": "field_center",
            "x_axis": "east_positive_long_axis",
            "y_axis": "north_positive_short_axis",
            "units": "meters",
        },
        "image_points": {
            label: [float(value) for value in state.image_points[label]]
            for label in labels
        },
        "world_points": {
            label: [float(value) for value in LANDMARKS[label]]
            for label in labels
        },
        "homography_image_to_world": homography.tolist(),
        "reprojection_errors_m": reprojection_errors,
        "reprojection_rmse_m": rmse,
    }

    camera_path = state.output_dir / f"camera_{state.camera_id}_calibration.json"
    with camera_path.open("w", encoding="utf-8") as output_file:
        json.dump(camera_payload, output_file, indent=2)

    combined_path = Path(config.CAMERA_IMAGE_TO_WORLD_HOMOGRAPHIES_PATH)
    combined = _load_combined_payload(combined_path)
    combined["field_length_m"] = FIELD_LENGTH_M
    combined["field_width_m"] = FIELD_WIDTH_M
    combined["world_coordinate_system"] = {
        "origin": "field_center",
        "x_axis": "east_positive_long_axis",
        "y_axis": "north_positive_short_axis",
        "units": "meters",
    }
    combined.setdefault("homographies", {})[str(state.camera_id)] = homography.tolist()
    combined.setdefault("camera_calibrations", {})[str(state.camera_id)] = camera_payload
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as output_file:
        json.dump(combined, output_file, indent=2)

    overlay = _build_canvas(state, for_display=False)
    overlay_path = state.output_dir / f"camera_{state.camera_id}_calibration_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Saved camera {state.camera_id} calibration:")
    print(f"  points={len(labels)} rmse={rmse:.4f} m")
    print(f"  camera_json={camera_path}")
    print(f"  homographies_json={combined_path}")
    print(f"  overlay={overlay_path}")


def _compute_homography(
    labels: Sequence[str],
    image_points: Dict[str, Tuple[float, float]],
) -> Tuple[np.ndarray, Dict[str, float], float]:
    src = np.asarray([image_points[label] for label in labels], dtype=np.float64)
    dst = np.asarray([LANDMARKS[label] for label in labels], dtype=np.float64)
    homography, _mask = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise RuntimeError("Failed to compute homography from selected points")

    projected = _project_points(homography, src)
    errors = np.linalg.norm(projected - dst, axis=1)
    reprojection_errors = {
        label: float(error)
        for label, error in zip(labels, errors)
    }
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return homography.astype(np.float64), reprojection_errors, rmse


def _project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    projected = homogeneous.dot(homography.T)
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    return projected[:, :2]


def _build_canvas(state: CalibrationState, *, for_display: bool) -> np.ndarray:
    canvas = state.image_bgr.copy()
    for label, point in state.image_points.items():
        color = (0, 255, 0) if label == state.selected_label else (0, 80, 255)
        _draw_point(canvas, label, point, color=color, scale=1.0)

    selected_world = LANDMARKS[state.selected_label]
    header_lines = [
        f"Camera {state.camera_id} | selected: {state.selected_label}",
        f"world xy = ({selected_world[0]:.2f}, {selected_world[1]:.2f}) m | clicked {len(state.image_points)} pts",
        "left-click set | n/space next | p prev | d delete | u undo | r reset | s save | q skip | esc quit",
        state.status,
    ]
    _draw_header(canvas, header_lines)

    if for_display and abs(state.scale - 1.0) > 1e-6:
        canvas = cv2.resize(canvas, None, fx=state.scale, fy=state.scale, interpolation=cv2.INTER_AREA)
    return canvas


def _draw_point(
    canvas: np.ndarray,
    label: str,
    point: Tuple[float, float],
    *,
    color: Tuple[int, int, int],
    scale: float,
) -> None:
    x = int(round(point[0] * scale))
    y = int(round(point[1] * scale))
    cv2.circle(canvas, (x, y), 7, color, thickness=-1)
    cv2.circle(canvas, (x, y), 11, (255, 255, 255), thickness=2)
    cv2.putText(
        canvas,
        label,
        (x + 12, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (x + 12, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_header(canvas: np.ndarray, lines: Sequence[str]) -> None:
    line_height = 24
    height = 14 + line_height * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0.0, canvas)
    for index, line in enumerate(lines):
        y = 24 + index * line_height
        color = (80, 255, 255) if index == 0 else (255, 255, 255)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def _display_scale(shape: Tuple[int, int, int], max_width: int, max_height: int) -> float:
    height, width = shape[:2]
    if max_width <= 0 or max_height <= 0:
        return 1.0
    return min(float(max_width) / float(width), float(max_height) / float(height), 1.0)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _load_existing_image_points(output_dir: Path, camera_id: int) -> Dict[str, Tuple[float, float]]:
    path = output_dir / f"camera_{camera_id}_calibration.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return {
        str(label): (float(point[0]), float(point[1]))
        for label, point in payload.get("image_points", {}).items()
        if label in LANDMARKS
    }


def _load_combined_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"homographies": {}, "camera_calibrations": {}}
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    payload.setdefault("homographies", {})
    payload.setdefault("camera_calibrations", {})
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
