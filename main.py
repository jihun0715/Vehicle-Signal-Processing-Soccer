"""ISSIA Soccer 추론 파이프라인 실행 엔트리포인트.

클래스는 없고, `main()`이 전체 흐름을 담당한다. 주요 기능은 config 기반으로
ISSIA synchronized dataloader를 만들고, `SoccerTrackingPipeline`을 통해
YOLO detection -> world projection -> Kalman tracking을 수행한 뒤 JSONL 결과와
카메라별 상하분할 tracking visualization video, frame별 profiling JSON을 저장하는 것이다.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import config
from data import create_issia_dataloader, discover_issia_reference_images
from model import SoccerTrackingPipeline, summarize_camera_result
from utils import TrackingVisualizationWriter


def main() -> int:
    end_frame = (
        config.PIPELINE_START_FRAME
        + max(config.PIPELINE_NUM_FRAMES - 1, 0) * config.PIPELINE_FRAME_STEP
    )

    print("Soccer tracking pipeline")
    print(f"  root={config.ISSIA_SOCCER_ROOT}")
    print(f"  cameras={config.PIPELINE_CAMERAS}")
    print(f"  yolo_model={config.YOLO_MODEL_PATH}")
    print(f"  frames={config.PIPELINE_START_FRAME}:{end_frame}:{config.PIPELINE_FRAME_STEP}")
    print(f"  output={config.PIPELINE_OUTPUT_PATH}")
    print(f"  profile_output={config.PIPELINE_PROFILE_OUTPUT_PATH}")
    if config.PIPELINE_SAVE_VISUALIZATION:
        print(f"  visualization_dir={config.PIPELINE_VIS_OUTPUT_DIR}")

    reference_images = discover_issia_reference_images(
        config.ISSIA_SOCCER_ROOT,
        config.PIPELINE_CAMERAS,
        pattern=config.ISSIA_REFERENCE_IMAGE_PATTERN,
    )
    if reference_images:
        refs = ", ".join(f"cam{camera_id}:{path.name}" for camera_id, path in reference_images.items())
        print(f"  reference_images={refs}")

    try:
        pipeline = SoccerTrackingPipeline.from_config()
    except ImportError as exc:
        print(f"Failed to initialize detector: {exc}")
        return 1

    dataloader = create_issia_dataloader(
        root=config.ISSIA_SOCCER_ROOT,
        cameras=config.PIPELINE_CAMERAS,
        synchronized=True,
        batch_size=config.PIPELINE_BATCH_SIZE,
        shuffle=False,
        num_workers=config.PIPELINE_NUM_WORKERS,
        pin_memory=config.DATALOADER_PIN_MEMORY,
        include_empty=True,
        require_all_cameras=True,
        frame_step=config.PIPELINE_FRAME_STEP,
        start_frame=config.PIPELINE_START_FRAME,
        end_frame=end_frame,
        image_mode="bgr",
        load_images=True,
        return_tensors=False,
    )

    visualizer = (
        TrackingVisualizationWriter.from_config()
        if config.PIPELINE_SAVE_VISUALIZATION
        else None
    )

    try:
        processed = 0
        should_stop = False
        batch_index = 0
        frame_profiles = []
        config.PIPELINE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.PIPELINE_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
            dataloader_iter = iter(dataloader)
            while processed < config.PIPELINE_NUM_FRAMES:
                dataloader_start = time.perf_counter()
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                dataloader_batch_sec = time.perf_counter() - dataloader_start
                samples = batch["samples"]
                dataloader_amortized_sec = dataloader_batch_sec / max(len(samples), 1)

                for sample_index, sample in enumerate(samples):
                    if processed >= config.PIPELINE_NUM_FRAMES:
                        should_stop = True
                        break

                    frame_start = time.perf_counter()
                    pipeline_profile: Dict[str, Any] = {}
                    result = pipeline.process_sample(sample, profile=pipeline_profile)

                    json_start = time.perf_counter()
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    json_write_sec = time.perf_counter() - json_start

                    visualization_sec = 0.0
                    if visualizer is not None:
                        visualization_start = time.perf_counter()
                        visualizer.write(sample, result)
                        visualization_sec = time.perf_counter() - visualization_start

                    frame_processing_sec = time.perf_counter() - frame_start
                    frame_profile = _build_frame_profile(
                        result,
                        pipeline_profile=pipeline_profile,
                        batch_index=batch_index,
                        sample_index=sample_index,
                        dataloader_batch_sec=dataloader_batch_sec,
                        dataloader_amortized_sec=dataloader_amortized_sec,
                        json_write_sec=json_write_sec,
                        visualization_sec=visualization_sec,
                        frame_processing_sec=frame_processing_sec,
                    )
                    frame_profiles.append(frame_profile)

                    summary = ", ".join(
                        summarize_camera_result(camera_result)
                        for _camera_id, camera_result in sorted(result["cameras"].items())
                    )
                    print(
                        f"frame={result['frame_index']} "
                        f"t={float(result['timestamp_sec']):.3f}s | {summary}"
                    )
                    processed += 1
                    if processed >= config.PIPELINE_NUM_FRAMES:
                        should_stop = True
                        break
                if should_stop:
                    break
                batch_index += 1
    finally:
        if visualizer is not None:
            visualizer.close()
        _close_dataset(getattr(dataloader, "dataset", None))
        _save_pipeline_profile(frame_profiles)

    print(f"Saved tracking results: {config.PIPELINE_OUTPUT_PATH}")
    print(f"Saved pipeline profile: {config.PIPELINE_PROFILE_OUTPUT_PATH}")
    if visualizer is not None:
        for camera_id, output_path in sorted(visualizer.output_paths.items()):
            print(f"Saved tracking visualization cam{camera_id}: {output_path}")
    return 0


def _build_frame_profile(
    result: Dict[str, Any],
    *,
    pipeline_profile: Dict[str, Any],
    batch_index: int,
    sample_index: int,
    dataloader_batch_sec: float,
    dataloader_amortized_sec: float,
    json_write_sec: float,
    visualization_sec: float,
    frame_processing_sec: float,
) -> Dict[str, Any]:
    cameras = pipeline_profile.get("cameras", {})
    aggregate = _sum_camera_profile(cameras)
    return {
        "frame_index": result.get("frame_index"),
        "timestamp_sec": result.get("timestamp_sec"),
        "batch_index": int(batch_index),
        "sample_index_in_batch": int(sample_index),
        "dataloader_batch_sec": float(dataloader_batch_sec),
        "dataloader_amortized_sec": float(dataloader_amortized_sec),
        "pipeline_total_sec": float(pipeline_profile.get("pipeline_total_sec", 0.0)),
        "json_write_sec": float(json_write_sec),
        "visualization_sec": float(visualization_sec),
        "frame_processing_sec": float(frame_processing_sec),
        "frame_total_with_dataloader_sec": float(frame_processing_sec + dataloader_amortized_sec),
        "aggregate": aggregate,
        "cameras": cameras,
    }


def _sum_camera_profile(cameras: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "detection_sec",
        "projection_sec",
        "kalman_sec",
        "camera_total_sec",
        "num_detections",
        "num_observations",
        "num_tracks",
    )
    aggregate = {key: 0.0 for key in keys}
    for camera_profile in cameras.values():
        for key in keys:
            aggregate[key] += float(camera_profile.get(key, 0.0))
    for key in ("num_detections", "num_observations", "num_tracks"):
        aggregate[key] = int(aggregate[key])
    return aggregate


def _save_pipeline_profile(frame_profiles: List[Dict[str, Any]]) -> None:
    config.PIPELINE_PROFILE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "root": str(config.ISSIA_SOCCER_ROOT),
            "cameras": list(config.PIPELINE_CAMERAS),
            "start_frame": config.PIPELINE_START_FRAME,
            "num_requested_frames": config.PIPELINE_NUM_FRAMES,
            "num_profiled_frames": len(frame_profiles),
            "frame_step": config.PIPELINE_FRAME_STEP,
            "batch_size": config.PIPELINE_BATCH_SIZE,
            "num_workers": config.PIPELINE_NUM_WORKERS,
            "tracking_output_path": str(config.PIPELINE_OUTPUT_PATH),
            "visualization_enabled": bool(config.PIPELINE_SAVE_VISUALIZATION),
        },
        "summary": _summarize_profiles(frame_profiles),
        "frames": frame_profiles,
    }
    with config.PIPELINE_PROFILE_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def _summarize_profiles(frame_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not frame_profiles:
        return {}

    keys = (
        "dataloader_amortized_sec",
        "pipeline_total_sec",
        "json_write_sec",
        "visualization_sec",
        "frame_processing_sec",
        "frame_total_with_dataloader_sec",
    )
    summary: Dict[str, Any] = {}
    for key in keys:
        values = [float(frame.get(key, 0.0)) for frame in frame_profiles]
        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    aggregate_keys = ("detection_sec", "projection_sec", "kalman_sec", "camera_total_sec")
    summary["aggregate"] = {}
    for key in aggregate_keys:
        values = [float(frame.get("aggregate", {}).get(key, 0.0)) for frame in frame_profiles]
        summary["aggregate"][key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _close_dataset(dataset: Any) -> None:
    if dataset is not None and hasattr(dataset, "close"):
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
