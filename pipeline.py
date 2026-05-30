"""End-to-end pieces for detection, projection, and per-camera tracking."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from detectors import YoloPersonDetector
from utils import ImageToWorldProjector, KalmanTrackerConfig, WorldKalmanTracker


class SoccerTrackingPipeline:
    """YOLO person detection -> world projection -> Kalman tracking."""

    def __init__(
        self,
        detector: YoloPersonDetector,
        projector: ImageToWorldProjector,
        tracker_config: KalmanTrackerConfig,
        *,
        cameras: Sequence[int] = (),
    ) -> None:
        self.detector = detector
        self.projector = projector
        self.tracker_config = tracker_config
        self.trackers: Dict[int, WorldKalmanTracker] = {
            int(camera_id): WorldKalmanTracker(tracker_config, camera_id=int(camera_id))
            for camera_id in cameras
        }

    @classmethod
    def from_config(cls) -> "SoccerTrackingPipeline":
        import config as project_config

        return cls(
            detector=YoloPersonDetector.from_config(),
            projector=ImageToWorldProjector.from_config(),
            tracker_config=KalmanTrackerConfig.from_config(),
            cameras=project_config.PIPELINE_CAMERAS,
        )

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset()

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "cameras" in sample:
            return self.process_sync_sample(sample)
        return self.process_camera_sample(sample)

    def process_sync_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        camera_results = {}
        for camera_id, camera_sample in sorted(sample["cameras"].items()):
            camera_results[int(camera_id)] = self.process_camera_sample(camera_sample)

        return {
            "frame_index": sample.get("frame_index"),
            "base_frame_index": sample.get("base_frame_index", sample.get("frame_index")),
            "timestamp_sec": sample.get("timestamp_sec"),
            "time_offset_gt": sample.get("time_offset_gt"),
            "cameras": camera_results,
        }

    def process_camera_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        meta = sample["meta"]
        camera_id = int(meta["camera_id"])
        frame_index = int(meta["frame_index"])
        timestamp_sec = float(meta.get("timestamp_sec", 0.0))
        image_size = _image_size_from_meta(meta)

        detections = self.detector.detect(sample["image"], meta=meta)
        observations = self.projector.detections_to_observations(
            detections,
            camera_id=camera_id,
            image_size=image_size,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
        )
        tracker = self._get_tracker(camera_id)
        tracks = tracker.update(observations, timestamp_sec=timestamp_sec)

        return {
            "camera_id": camera_id,
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "detections": [detection.to_dict() for detection in detections],
            "observations": [observation.to_dict() for observation in observations],
            "tracks": [track.to_dict() for track in tracks],
            "confirmed_tracks": [
                track.to_dict() for track in tracker.get_tracks(confirmed_only=True)
            ],
        }

    def _get_tracker(self, camera_id: int) -> WorldKalmanTracker:
        camera_id = int(camera_id)
        if camera_id not in self.trackers:
            self.trackers[camera_id] = WorldKalmanTracker(self.tracker_config, camera_id=camera_id)
        return self.trackers[camera_id]


def summarize_camera_result(camera_result: Dict[str, Any]) -> str:
    """Compact text summary for console debugging."""

    camera_id = camera_result["camera_id"]
    return (
        f"cam{camera_id}: "
        f"det={len(camera_result['detections'])}, "
        f"obs={len(camera_result['observations'])}, "
        f"tracks={len(camera_result['tracks'])}, "
        f"confirmed={len(camera_result['confirmed_tracks'])}"
    )


def _image_size_from_meta(meta: Dict[str, Any]) -> Tuple[int, int]:
    image_size = meta.get("image_size")
    if image_size is None:
        raise KeyError("sample meta is missing image_size")
    return (int(image_size[0]), int(image_size[1]))
