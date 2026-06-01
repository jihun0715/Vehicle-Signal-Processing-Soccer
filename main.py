"""Run YOLO -> projection -> Kalman tracking on ISSIA Soccer frames.

Defaults are intentionally read from config.py so local paths, model weights,
and tracker parameters can be changed in one place.
"""

from __future__ import annotations

import json
from typing import Any

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
        config.PIPELINE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.PIPELINE_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
            for batch in dataloader:
                for sample in batch["samples"]:
                    result = pipeline.process_sample(sample)
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if visualizer is not None:
                        visualizer.write(sample, result)

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
    finally:
        if visualizer is not None:
            visualizer.close()
        _close_dataset(getattr(dataloader, "dataset", None))

    print(f"Saved tracking results: {config.PIPELINE_OUTPUT_PATH}")
    if visualizer is not None:
        for camera_id, output_path in sorted(visualizer.output_paths.items()):
            print(f"Saved tracking visualization cam{camera_id}: {output_path}")
    return 0


def _close_dataset(dataset: Any) -> None:
    if dataset is not None and hasattr(dataset, "close"):
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
