"""PyTorch datasets for the ISSIA-CNR soccer dataset.

The local ISSIA-Soccer dump is expected to keep the original layout:

    ISSIA-Soccer/
    |-- Annotation/
    |   `-- Film Role-0 ID-1 ... .xgtf
    `-- Sequences/
        `-- Film Role-0 ID-1 ... .avi

The returned samples keep detection metadata as first-class values because this
project uses the data as an input stream for detection, projection, tracking,
track matching, pose estimation, and temporal calibration rather than as a
single supervised training dataset.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # Allows parsing utilities to be used on hosts without torch.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

    DataLoader = None  # type: ignore[assignment]


PERSON_LABEL = 0
BALL_LABEL = 1
LABEL_NAMES = {
    PERSON_LABEL: "person",
    BALL_LABEL: "ball",
}

VIPER_NS = {
    "v": "http://lamp.cfar.umd.edu/viper#",
    "data": "http://lamp.cfar.umd.edu/viperdata#",
}

DEFAULT_ISSIA_ROOT = Path(os.environ.get("ISSIA_SOCCER_ROOT", "/datasets/ISSIA-Soccer"))


@dataclass(frozen=True)
class ISSIADetection:
    """One parsed detection annotation in image coordinates."""

    camera_id: int
    frame_index: int
    label: str
    label_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    object_id: int
    point_xy: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "label": self.label,
            "label_id": self.label_id,
            "bbox_xyxy": self.bbox_xyxy,
            "object_id": self.object_id,
            "point_xy": self.point_xy,
        }


@dataclass(frozen=True)
class ISSIAFrameRecord:
    """Index entry for one camera frame."""

    dataset_root: Path
    camera_id: int
    frame_index: int
    timestamp_sec: float
    video_path: Path
    image_size: Tuple[int, int]
    detections: Tuple[ISSIADetection, ...]


PathLikeStr = Union[str, os.PathLike]


def discover_issia_cameras(root: PathLikeStr = DEFAULT_ISSIA_ROOT) -> List[int]:
    """Return camera IDs discovered from the ISSIA annotation files."""

    root_path = Path(root)
    annotation_dir = root_path / "Annotation"
    camera_ids = set()
    for path in annotation_dir.glob("*.xgtf"):
        camera_id = _camera_id_from_path(path)
        if camera_id is not None:
            camera_ids.add(camera_id)
    return sorted(camera_ids)


def read_issia_annotations(
    root: PathLikeStr,
    camera_id: int,
    *,
    ball_bbox_size: int = 20,
    image_size: Optional[Tuple[int, int]] = None,
    clip_boxes: bool = True,
) -> Dict[int, List[ISSIADetection]]:
    """Parse a camera xgtf file into frame-indexed detections.

    Args:
        root: ISSIA-Soccer dataset root.
        camera_id: Camera ID between 1 and 6.
        ball_bbox_size: Pixel size of the square pseudo-bbox built around the
            annotated ball center.
        image_size: Optional ``(height, width)`` used for bbox clipping.
        clip_boxes: Clip boxes to image bounds and drop fully invalid boxes.
    """

    root_path = Path(root)
    annotation_path = _find_camera_file(root_path / "Annotation", camera_id, ".xgtf")
    frame_to_detections: Dict[int, List[ISSIADetection]] = {}
    tree = ET.parse(annotation_path)
    root_node = tree.getroot()

    for obj in root_node.findall(".//v:object", VIPER_NS):
        obj_name = obj.attrib.get("name", "")
        object_id = _safe_int(obj.attrib.get("id"), default=-1)

        if obj_name == "Person":
            _append_person_annotations(
                obj,
                frame_to_detections,
                camera_id,
                object_id,
                image_size=image_size,
                clip_boxes=clip_boxes,
            )
        elif obj_name == "BALL":
            _append_ball_annotations(
                obj,
                frame_to_detections,
                camera_id,
                object_id,
                ball_bbox_size=ball_bbox_size,
                image_size=image_size,
                clip_boxes=clip_boxes,
            )

    return frame_to_detections


class ISSIASoccerFrameDataset(Dataset):
    """Frame-level ISSIA dataset for one or more cameras.

    Each item is a dictionary:

    ``{"image": image, "target": target, "meta": meta}``

    ``target["boxes"]`` uses xyxy pixel coordinates. The labels are internal
    project labels: ``0=person`` and ``1=ball``. Object IDs come from the VIPER
    annotation object IDs; the ball object ID is usually 0.
    """

    def __init__(
        self,
        root: PathLikeStr = DEFAULT_ISSIA_ROOT,
        cameras: Sequence[int] = (1, 2, 3, 4, 5, 6),
        *,
        include_empty: bool = False,
        only_ball_frames: bool = False,
        frame_step: int = 1,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        ball_bbox_size: int = 20,
        image_mode: str = "rgb",
        load_images: bool = True,
        return_tensors: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        if image_mode not in {"rgb", "bgr"}:
            raise ValueError("image_mode must be 'rgb' or 'bgr'")
        if return_tensors and torch is None:
            raise ImportError("PyTorch is required when return_tensors=True")

        self.root = Path(root)
        self.cameras = tuple(int(camera_id) for camera_id in cameras)
        self.include_empty = include_empty
        self.only_ball_frames = only_ball_frames
        self.frame_step = frame_step
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.ball_bbox_size = ball_bbox_size
        self.image_mode = image_mode
        self.load_images = load_images
        self.return_tensors = return_tensors
        self.transform = transform
        self._captures: Dict[int, cv2.VideoCapture] = {}

        self.records = self._build_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = self._read_image(record) if self.load_images else None
        target = self._build_target(record)
        meta = {
            "dataset_root": str(record.dataset_root),
            "video_path": str(record.video_path),
            "camera_id": record.camera_id,
            "frame_index": record.frame_index,
            "timestamp_sec": record.timestamp_sec,
            "image_size": record.image_size,
        }

        sample = {
            "image": image,
            "target": target,
            "meta": meta,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_captures"] = {}
        return state

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def _build_records(self) -> List[ISSIAFrameRecord]:
        if not self.root.exists():
            raise FileNotFoundError(f"Cannot find ISSIA dataset root: {self.root}")

        records: List[ISSIAFrameRecord] = []
        for camera_id in self.cameras:
            video_path = _find_camera_file(self.root / "Sequences", camera_id, ".avi")
            video_info = _read_video_info(video_path)
            image_size = (video_info["height"], video_info["width"])
            annotations = read_issia_annotations(
                self.root,
                camera_id,
                ball_bbox_size=self.ball_bbox_size,
                image_size=image_size,
            )

            if self.include_empty:
                frame_ids = list(range(video_info["num_frames"]))
            else:
                frame_ids = sorted(annotations.keys())

            if self.only_ball_frames:
                frame_ids = [
                    frame_id
                    for frame_id in frame_ids
                    if any(det.label_id == BALL_LABEL for det in annotations.get(frame_id, []))
                ]

            for frame_id in frame_ids:
                if frame_id < 0 or frame_id >= video_info["num_frames"]:
                    continue
                if self.start_frame is not None and frame_id < self.start_frame:
                    continue
                if self.end_frame is not None and frame_id > self.end_frame:
                    continue
                if (frame_id - (self.start_frame or 0)) % self.frame_step != 0:
                    continue

                timestamp_sec = frame_id / video_info["fps"] if video_info["fps"] > 0 else 0.0
                records.append(
                    ISSIAFrameRecord(
                        dataset_root=self.root,
                        camera_id=camera_id,
                        frame_index=frame_id,
                        timestamp_sec=timestamp_sec,
                        video_path=video_path,
                        image_size=image_size,
                        detections=tuple(annotations.get(frame_id, [])),
                    )
                )

        records.sort(key=lambda rec: (rec.frame_index, rec.camera_id))
        return records

    def _build_target(self, record: ISSIAFrameRecord) -> Dict[str, Any]:
        boxes = np.asarray([det.bbox_xyxy for det in record.detections], dtype=np.float32)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)

        labels = np.asarray([det.label_id for det in record.detections], dtype=np.int64)
        object_ids = np.asarray([det.object_id for det in record.detections], dtype=np.int64)
        is_ball = labels == BALL_LABEL

        if self.return_tensors:
            boxes_out = torch.from_numpy(boxes)
            labels_out = torch.from_numpy(labels)
            object_ids_out = torch.from_numpy(object_ids)
            is_ball_out = torch.from_numpy(is_ball)
        else:
            boxes_out = boxes
            labels_out = labels
            object_ids_out = object_ids
            is_ball_out = is_ball

        return {
            "boxes": boxes_out,
            "labels": labels_out,
            "object_ids": object_ids_out,
            "is_ball": is_ball_out,
            "detections": [det.to_dict() for det in record.detections],
            "camera_id": record.camera_id,
            "frame_index": record.frame_index,
            "timestamp_sec": record.timestamp_sec,
            "image_size": record.image_size,
            "label_names": LABEL_NAMES,
        }

    def _read_image(self, record: ISSIAFrameRecord) -> Any:
        capture = self._get_capture(record.camera_id, record.video_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, record.frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to read camera {record.camera_id} frame {record.frame_index} "
                f"from {record.video_path}"
            )
        if self.image_mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.return_tensors:
            return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        return frame

    def _get_capture(self, camera_id: int, video_path: Path) -> cv2.VideoCapture:
        capture = self._captures.get(camera_id)
        if capture is None or not capture.isOpened():
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self._captures[camera_id] = capture
        return capture


class ISSIASoccerSyncDataset(Dataset):
    """Synchronized multi-camera view over ``ISSIASoccerFrameDataset``."""

    def __init__(
        self,
        root: PathLikeStr = DEFAULT_ISSIA_ROOT,
        cameras: Sequence[int] = (1, 2, 3, 4, 5, 6),
        *,
        require_all_cameras: bool = True,
        **frame_dataset_kwargs: Any,
    ) -> None:
        self.frame_dataset = ISSIASoccerFrameDataset(root, cameras, **frame_dataset_kwargs)
        self.cameras = tuple(int(camera_id) for camera_id in cameras)
        self.require_all_cameras = require_all_cameras
        self._index_by_camera_frame = {
            (record.camera_id, record.frame_index): idx
            for idx, record in enumerate(self.frame_dataset.records)
        }
        self.frame_indices = self._build_frame_indices()

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame_index = self.frame_indices[index]
        camera_samples = {}
        timestamps = []
        for camera_id in self.cameras:
            sample_index = self._index_by_camera_frame.get((camera_id, frame_index))
            if sample_index is None:
                if self.require_all_cameras:
                    raise KeyError(f"Missing camera {camera_id} for frame {frame_index}")
                continue
            sample = self.frame_dataset[sample_index]
            camera_samples[camera_id] = sample
            timestamps.append(sample["meta"]["timestamp_sec"])

        timestamp_sec = float(np.mean(timestamps)) if timestamps else 0.0
        return {
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "cameras": camera_samples,
        }

    def close(self) -> None:
        self.frame_dataset.close()

    def _build_frame_indices(self) -> List[int]:
        frame_sets = []
        for camera_id in self.cameras:
            frames = {
                record.frame_index
                for record in self.frame_dataset.records
                if record.camera_id == camera_id
            }
            frame_sets.append(frames)

        if not frame_sets:
            return []

        if self.require_all_cameras:
            selected = set.intersection(*frame_sets)
        else:
            selected = set.union(*frame_sets)
        return sorted(selected)


def collate_issia_samples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate ISSIA samples without stacking variable-length targets."""

    if not batch:
        return {}

    if "cameras" in batch[0]:
        camera_ids = sorted({camera_id for item in batch for camera_id in item["cameras"]})
        camera_batches = {}
        for camera_id in camera_ids:
            samples = [item["cameras"][camera_id] for item in batch if camera_id in item["cameras"]]
            camera_batches[camera_id] = collate_issia_samples(samples)
        return {
            "frame_indices": [item["frame_index"] for item in batch],
            "timestamp_sec": [item["timestamp_sec"] for item in batch],
            "camera_batches": camera_batches,
            "samples": batch,
        }

    return {
        "images": [item["image"] for item in batch],
        "targets": [item["target"] for item in batch],
        "metas": [item["meta"] for item in batch],
        "samples": batch,
    }


def create_issia_dataloader(
    root: PathLikeStr = DEFAULT_ISSIA_ROOT,
    cameras: Sequence[int] = (1, 2, 3, 4, 5, 6),
    *,
    synchronized: bool = False,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    **dataset_kwargs: Any,
) -> Any:
    """Create a DataLoader for ISSIA frame or synchronized-camera samples."""

    if DataLoader is None:
        raise ImportError("PyTorch is required to create a DataLoader")

    if synchronized:
        dataset = ISSIASoccerSyncDataset(root, cameras, **dataset_kwargs)
    else:
        dataset = ISSIASoccerFrameDataset(root, cameras, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_issia_samples,
    )


def _append_person_annotations(
    obj: ET.Element,
    frame_to_detections: Dict[int, List[ISSIADetection]],
    camera_id: int,
    object_id: int,
    *,
    image_size: Optional[Tuple[int, int]],
    clip_boxes: bool,
) -> None:
    location_attr = _find_attribute(obj, "LOCATION")
    if location_attr is None:
        return

    for bbox_node in location_attr.findall("data:bbox", VIPER_NS):
        x1 = float(bbox_node.attrib["x"])
        y1 = float(bbox_node.attrib["y"])
        width = float(bbox_node.attrib["width"])
        height = float(bbox_node.attrib["height"])
        bbox = (x1, y1, x1 + width, y1 + height)
        bbox = _clip_or_none(bbox, image_size) if clip_boxes else bbox
        if bbox is None:
            continue
        for frame_id in _iter_framespan(bbox_node.attrib.get("framespan", "")):
            frame_to_detections.setdefault(frame_id, []).append(
                ISSIADetection(
                    camera_id=camera_id,
                    frame_index=frame_id,
                    label="person",
                    label_id=PERSON_LABEL,
                    bbox_xyxy=bbox,
                    object_id=object_id,
                )
            )


def _append_ball_annotations(
    obj: ET.Element,
    frame_to_detections: Dict[int, List[ISSIADetection]],
    camera_id: int,
    object_id: int,
    *,
    ball_bbox_size: int,
    image_size: Optional[Tuple[int, int]],
    clip_boxes: bool,
) -> None:
    ball_attr = _find_attribute(obj, "BallPos")
    if ball_attr is None:
        return

    half = float(ball_bbox_size) / 2.0
    for point_node in ball_attr.findall("data:point", VIPER_NS):
        x = float(point_node.attrib["x"])
        y = float(point_node.attrib["y"])
        bbox = (x - half, y - half, x + half, y + half)
        bbox = _clip_or_none(bbox, image_size) if clip_boxes else bbox
        if bbox is None:
            continue
        for frame_id in _iter_framespan(point_node.attrib.get("framespan", "")):
            frame_to_detections.setdefault(frame_id, []).append(
                ISSIADetection(
                    camera_id=camera_id,
                    frame_index=frame_id,
                    label="ball",
                    label_id=BALL_LABEL,
                    bbox_xyxy=bbox,
                    object_id=object_id,
                    point_xy=(x, y),
                )
            )


def _find_attribute(obj: ET.Element, name: str) -> Optional[ET.Element]:
    for attr in obj.findall("v:attribute", VIPER_NS):
        if attr.attrib.get("name") == name:
            return attr
    return None


def _iter_framespan(framespan: str) -> Iterable[int]:
    for part in framespan.replace(",", " ").split():
        if not part:
            continue
        if ":" in part:
            start_s, end_s = part.split(":", 1)
            start = int(start_s)
            end = int(end_s)
        else:
            start = end = int(part)
        if end < start:
            start, end = end, start
        yield from range(start, end + 1)


def _clip_or_none(
    bbox_xyxy: Tuple[float, float, float, float],
    image_size: Optional[Tuple[int, int]],
) -> Optional[Tuple[float, float, float, float]]:
    if image_size is None:
        return bbox_xyxy

    height, width = image_size
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _read_video_info(video_path: Path) -> Dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    info = {
        "num_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return info


def _find_camera_file(directory: Path, camera_id: int, suffix: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Missing directory: {directory}")

    pattern = f"*ID-{camera_id}*{suffix}"
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Cannot find camera {camera_id} file with pattern {pattern}")
    if len(matches) > 1:
        exact_matches = [path for path in matches if _camera_id_from_path(path) == camera_id]
        if len(exact_matches) == 1:
            return exact_matches[0]
    return matches[0]


def _camera_id_from_path(path: Path) -> Optional[int]:
    match = re.search(r"ID-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _safe_int(value: Optional[str], default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default
