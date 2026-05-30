"""Quick checks for the ISSIA soccer dataloader.

Examples:
    python debug.py
    python debug.py --root /home/jihun/Documents/ISSIA-Soccer
    python debug.py --cameras 1 2 --frame-step 250 --save-preview
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from config import DEBUG_OUTPUT_DIR, ISSIA_CAMERAS, ISSIA_SOCCER_ROOT
from data import (
    BALL_LABEL,
    ISSIASoccerFrameDataset,
    ISSIASoccerSyncDataset,
    create_issia_dataloader,
    discover_issia_cameras,
    read_issia_annotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug ISSIA dataset/dataloader wiring.")
    parser.add_argument("--root", type=Path, default=ISSIA_SOCCER_ROOT, help="ISSIA-Soccer root path.")
    parser.add_argument("--cameras", type=int, nargs="+", default=list(ISSIA_CAMERAS), help="Camera IDs to inspect.")
    parser.add_argument("--frame-step", type=int, default=500, help="Subsample frames for quick checks.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for DataLoader check.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--save-preview", action="store_true", help="Save one bbox overlay image.")
    parser.add_argument("--output-dir", type=Path, default=DEBUG_OUTPUT_DIR, help="Preview output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            frame_step=args.frame_step,
            load_images=False,
            return_tensors=False,
        )
        batch = next(iter(loader))
        print(f"Batch image slots: {len(batch['images'])}")
        print(f"Batch metas: {[short_meta(meta) for meta in batch['metas']]}")
    except ImportError as exc:
        print(f"Skipped DataLoader check because PyTorch is unavailable: {exc}")

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
