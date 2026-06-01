"""YOLO detection, world projection, Kalman tracking을 연결하는 모델 파이프라인.

주요 클래스:
- `SoccerTrackingPipeline`: 카메라별 YOLO person detection 결과를 world 좌표계 observation으로 바꾸고,
  카메라별 `WorldKalmanTracker`를 갱신해 tracking 결과 dict를 만든다.

주요 함수:
- `summarize_camera_result`: 콘솔 로그에 출력하기 좋은 카메라별 detection/tracking 요약 문자열을 만든다.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

from model.yolo_detector import YoloPersonDetector
from utils.kalman_filter import KalmanTrackerConfig, WorldKalmanTracker
from utils.projection import ImageToWorldProjector


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

    def process_sample(
        self,
        sample: Dict[str, Any],
        *,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        if "cameras" in sample:
            result = self.process_sync_sample(sample, profile=profile)
        else:
            result = self.process_camera_sample(sample, profile=profile)
        if profile is not None:
            profile["pipeline_total_sec"] = time.perf_counter() - start
        return result

    def process_sync_sample(
        self,
        sample: Dict[str, Any],
        *,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        camera_results = {}
        camera_profiles = profile.setdefault("cameras", {}) if profile is not None else None
        for camera_id, camera_sample in sorted(sample["cameras"].items()):
            camera_profile = {} if camera_profiles is not None else None
            camera_results[int(camera_id)] = self.process_camera_sample(
                camera_sample,
                profile=camera_profile,
            )
            if camera_profiles is not None:
                camera_profiles[str(int(camera_id))] = camera_profile

        return {
            "frame_index": sample.get("frame_index"),
            "base_frame_index": sample.get("base_frame_index", sample.get("frame_index")),
            "timestamp_sec": sample.get("timestamp_sec"),
            "time_offset_gt": sample.get("time_offset_gt"),
            "cameras": camera_results,
        }

    def process_camera_sample(
        self,
        sample: Dict[str, Any],
        *,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        camera_start = time.perf_counter()
        meta = sample["meta"]
        camera_id = int(meta["camera_id"])
        frame_index = int(meta["frame_index"])
        timestamp_sec = float(meta.get("timestamp_sec", 0.0))
        image_size = _image_size_from_meta(meta)

        detection_start = time.perf_counter()
        detections = self.detector.detect(sample["image"], meta=meta)
        detection_sec = time.perf_counter() - detection_start

        projection_start = time.perf_counter()
        observations = self.projector.detections_to_observations(
            detections,
            camera_id=camera_id,
            image_size=image_size,
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
        )
        projection_sec = time.perf_counter() - projection_start

        tracking_start = time.perf_counter()
        tracker = self._get_tracker(camera_id)
        tracks = tracker.update(observations, timestamp_sec=timestamp_sec)
        kalman_sec = time.perf_counter() - tracking_start

        if profile is not None:
            profile.update(
                {
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "detection_sec": detection_sec,
                    "projection_sec": projection_sec,
                    "kalman_sec": kalman_sec,
                    "camera_total_sec": time.perf_counter() - camera_start,
                    "num_detections": len(detections),
                    "num_observations": len(observations),
                    "num_tracks": len(tracks),
                }
            )

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
