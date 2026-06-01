"""YOLO detection, world projection, Kalman tracking을 연결하는 모델 파이프라인.

주요 클래스:
- `SoccerTrackingPipeline`: 여러 카메라 이미지를 YOLO batch detection으로 묶어 처리한 뒤,
  카메라별 detection 결과를 world 좌표계 observation으로 바꾸고 `WorldKalmanTracker`를 갱신한다.

주요 함수:
- `summarize_camera_result`: 콘솔 로그에 출력하기 좋은 카메라별 detection/tracking 요약 문자열을 만든다.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model.yolo_detector import PersonDetection, YoloPersonDetector
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
        batch_detection: bool = True,
        batch_fallback_to_single: bool = True,
    ) -> None:
        self.detector = detector
        self.projector = projector
        self.tracker_config = tracker_config
        self.batch_detection = bool(batch_detection)
        self.batch_fallback_to_single = bool(batch_fallback_to_single)
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
            batch_detection=getattr(project_config, "YOLO_BATCH_INFERENCE", True),
            batch_fallback_to_single=getattr(
                project_config,
                "YOLO_BATCH_FALLBACK_TO_SINGLE",
                True,
            ),
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
        camera_items = [
            (int(camera_id), camera_sample)
            for camera_id, camera_sample in sorted(sample.get("cameras", {}).items())
        ]

        if (
            self.batch_detection
            and camera_items
            and hasattr(self.detector, "detect_batch")
        ):
            detection_start = time.perf_counter()
            batch_detections = self.detector.detect_batch(
                [camera_sample["image"] for _, camera_sample in camera_items],
                metas=[camera_sample.get("meta") for _, camera_sample in camera_items],
                fallback_to_single=self.batch_fallback_to_single,
            )
            batch_detection_sec = time.perf_counter() - detection_start
            if len(batch_detections) != len(camera_items):
                batch_detections = _normalize_batch_detections(
                    batch_detections,
                    len(camera_items),
                )
            per_camera_detection_sec = batch_detection_sec / max(len(camera_items), 1)

            if profile is not None:
                profile["batch_detection_enabled"] = True
                profile["batch_detection_sec"] = batch_detection_sec
                profile["batch_detection_num_images"] = len(camera_items)

            for (camera_id, camera_sample), detections in zip(camera_items, batch_detections):
                camera_profile = {} if camera_profiles is not None else None
                camera_results[int(camera_id)] = self.process_camera_sample(
                    camera_sample,
                    detections=detections,
                    detection_sec=per_camera_detection_sec,
                    detection_was_batched=True,
                    profile=camera_profile,
                )
                if camera_profiles is not None:
                    camera_profiles[str(int(camera_id))] = camera_profile
        else:
            if profile is not None:
                profile["batch_detection_enabled"] = False
                profile["batch_detection_sec"] = 0.0
                profile["batch_detection_num_images"] = 0

            for camera_id, camera_sample in camera_items:
                camera_profile = {} if camera_profiles is not None else None
                camera_results[int(camera_id)] = self.process_camera_sample(
                    camera_sample,
                    profile=camera_profile,
                )
                if camera_profiles is not None:
                    camera_profiles[str(int(camera_id))] = camera_profile

        if profile is not None and not camera_items:
            profile["batch_detection_enabled"] = False
            profile["batch_detection_sec"] = 0.0
            profile["batch_detection_num_images"] = 0

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
        detections: Optional[Sequence[PersonDetection]] = None,
        detection_sec: Optional[float] = None,
        detection_was_batched: bool = False,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        camera_start = time.perf_counter()
        meta = sample["meta"]
        camera_id = int(meta["camera_id"])
        frame_index = int(meta["frame_index"])
        timestamp_sec = float(meta.get("timestamp_sec", 0.0))
        image_size = _image_size_from_meta(meta)

        if detections is None:
            detection_start = time.perf_counter()
            detections_list = self.detector.detect(sample["image"], meta=meta)
            measured_detection_sec = time.perf_counter() - detection_start
        else:
            detections_list = list(detections)
            measured_detection_sec = float(detection_sec or 0.0)

        projection_start = time.perf_counter()
        observations = self.projector.detections_to_observations(
            detections_list,
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
            camera_total_sec = time.perf_counter() - camera_start
            if detection_was_batched:
                camera_total_sec += measured_detection_sec
            profile.update(
                {
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "detection_sec": measured_detection_sec,
                    "detection_batch_amortized": bool(detection_was_batched),
                    "projection_sec": projection_sec,
                    "kalman_sec": kalman_sec,
                    "camera_total_sec": camera_total_sec,
                    "num_detections": len(detections_list),
                    "num_observations": len(observations),
                    "num_tracks": len(tracks),
                }
            )

        return {
            "camera_id": camera_id,
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "detections": [detection.to_dict() for detection in detections_list],
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


def _normalize_batch_detections(
    batch_detections: Sequence[Sequence[PersonDetection]],
    target_length: int,
) -> List[List[PersonDetection]]:
    normalized = [list(detections) for detections in batch_detections[:target_length]]
    if len(normalized) < target_length:
        normalized.extend([] for _ in range(target_length - len(normalized)))
    return normalized


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
