"""Quick checks for the ISSIA soccer dataloader.

Example:
    python debug.py
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any, Dict

import cv2
import numpy as np

from config import (
    DEBUG_CAMERAS,
    DEBUG_FRAME_STEP,
    DEBUG_OUTPUT_DIR,
    DEBUG_SAVE_PREVIEW,
    DATALOADER_BATCH_SIZE,
    DATALOADER_NUM_WORKERS,
    DATALOADER_PIN_MEMORY,
    ISSIA_SOCCER_ROOT,
    OFFSET_MAX_FRAME_OFFSET,
    OFFSET_RANDOM_SEED,
)
from data import (
    BALL_LABEL,
    ISSIASoccerFrameDataset,
    ISSIASoccerSyncDataset,
    create_issia_dataloader,
    create_issia_offset_dataloader,
    discover_issia_cameras,
    read_issia_annotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug ISSIA dataset/dataloader wiring. Defaults come from config.py."
    )
    parser.add_argument("--save-preview", action="store_true", default=None, help="Save one bbox overlay image.")
    return parser.parse_args()


def load_settings(cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        root=ISSIA_SOCCER_ROOT,
        cameras=tuple(DEBUG_CAMERAS),
        frame_step=DEBUG_FRAME_STEP,
        batch_size=DATALOADER_BATCH_SIZE,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=DATALOADER_PIN_MEMORY,
        save_preview=DEBUG_SAVE_PREVIEW if cli_args.save_preview is None else cli_args.save_preview,
        output_dir=DEBUG_OUTPUT_DIR,
        offset_max_frame_offset=OFFSET_MAX_FRAME_OFFSET,
        offset_random_seed=OFFSET_RANDOM_SEED,
    )


def main() -> None:
    args = load_settings(parse_args())
    root = args.root.expanduser()
    cameras = tuple(args.cameras)

    print_header("Config")
    print(f"ISSIA root: {root}")
    print(f"Cameras: {cameras}")
    print(f"Frame step: {args.frame_step}")

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}\n"
            "Edit config.py, set ISSIA_SOCCER_ROOT, or pass --root."
        )

    print_header("Camera Discovery")
    discovered = discover_issia_cameras(root)
    print(f"Discovered cameras: {discovered}")

    print_header("Annotation Summary")
    for camera_id in cameras:
        annotations = read_issia_annotations(root, camera_id)
        frame_count = len(annotations)
        det_count = sum(len(detections) for detections in annotations.values())
        ball_frame_count = sum(
            1
            for detections in annotations.values()
            if any(det.label_id == BALL_LABEL for det in detections)
        )
        first_frame = min(annotations) if annotations else None
        last_frame = max(annotations) if annotations else None
        print(
            f"cam {camera_id}: annotated_frames={frame_count}, detections={det_count}, "
            f"ball_frames={ball_frame_count}, frame_range=({first_frame}, {last_frame})"
        )

    print_header("Frame Dataset")
    frame_dataset = ISSIASoccerFrameDataset(
        root,
        cameras=cameras,
        frame_step=args.frame_step,
        load_images=False,
        return_tensors=False,
    )
    print(f"Frame dataset samples: {len(frame_dataset)}")
    if len(frame_dataset) == 0:
        raise RuntimeError("Frame dataset is empty. Check cameras, frame range, and dataset root.")
    print_sample(frame_dataset[0])

    print_header("Image Read")
    image_dataset = ISSIASoccerFrameDataset(
        root,
        cameras=(cameras[0],),
        start_frame=frame_dataset[0]["meta"]["frame_index"],
        end_frame=frame_dataset[0]["meta"]["frame_index"],
        load_images=True,
        return_tensors=False,
    )
    image_sample = image_dataset[0]
    image = image_sample["image"]
    print(f"Loaded image shape: {image.shape}, dtype={image.dtype}")

    if args.save_preview:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = args.output_dir / (
            f"issia_cam{image_sample['meta']['camera_id']}_"
            f"frame{image_sample['meta']['frame_index']}.jpg"
        )
        save_preview(image_sample, preview_path)
        print(f"Saved preview: {preview_path}")
    image_dataset.close()

    print_header("Synchronized Dataset")
    sync_dataset = ISSIASoccerSyncDataset(
        root,
        cameras=cameras[: min(2, len(cameras))],
        frame_step=args.frame_step,
        load_images=False,
        return_tensors=False,
    )
    print(f"Sync dataset samples: {len(sync_dataset)}")
    if len(sync_dataset) == 0:
        raise RuntimeError("Synchronized dataset is empty. Try fewer cameras or a smaller frame step.")
    sync_sample = sync_dataset[0]
    print(
        f"First sync frame={sync_sample['frame_index']}, "
        f"cameras={sorted(sync_sample['cameras'].keys())}"
    )

    print_header("DataLoader")
    try:
        loader = create_issia_dataloader(
            root,
            cameras=cameras[: min(2, len(cameras))],
            synchronized=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            frame_step=args.frame_step,
            load_images=False,
            return_tensors=False,
        )
        batch = next(iter(loader))
        print(f"Batch image slots: {len(batch['images'])}")
        print(f"Batch metas: {[short_meta(meta) for meta in batch['metas']]}")
    except ImportError as exc:
        print(f"Skipped DataLoader check because PyTorch is unavailable: {exc}")

    print_header("Offset DataLoader")
    try:
        offset_loader = create_issia_offset_dataloader(
            root,
            cameras=cameras[: min(2, len(cameras))],
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            frame_step=args.frame_step,
            load_images=False,
            return_tensors=False,
            include_empty=True,
            start_frame=500,
            end_frame=500 + args.frame_step,
            max_frame_offset=args.offset_max_frame_offset,
            random_seed=args.offset_random_seed,
        )
        offset_batch = next(iter(offset_loader))
        print(f"Offset batch base frames: {offset_batch['base_frame_indices']}")
        print(f"Offset GT: {offset_batch['time_offset_gt'][0]}")
    except ImportError as exc:
        print(f"Skipped offset DataLoader check because PyTorch is unavailable: {exc}")

    frame_dataset.close()
    sync_dataset.close()


def print_header(title: str) -> None:
    print(f"\n== {title} ==")


def print_sample(sample: Dict[str, Any]) -> None:
    meta = sample["meta"]
    target = sample["target"]
    labels = target["labels"]
    boxes = target["boxes"]
    print(f"First sample meta: {short_meta(meta)}")
    print(f"Boxes shape: {boxes.shape}, labels: {labels[:10].tolist()}")


def short_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cam": meta["camera_id"],
        "frame": meta["frame_index"],
        "t": round(meta["timestamp_sec"], 3),
        "size": meta["image_size"],
    }


def save_preview(sample: Dict[str, Any], output_path: Path) -> None:
    image_rgb = sample["image"]
    target = sample["target"]
    preview = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    boxes = target["boxes"]
    labels = target["labels"]

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (0, 255, 0) if int(label) != BALL_LABEL else (0, 0, 255)
        text = "person" if int(label) != BALL_LABEL else "ball"
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), preview)


if __name__ == "__main__":
    main()
