"""Create a quick top/bottom video for temporally shifted camera streams.

Example:
    python debug_video.py --cameras 1 2 --max-offset 80 --seed 7 --num-frames 150
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np

from config import DEBUG_OUTPUT_DIR, ISSIA_SOCCER_ROOT
from data import BALL_LABEL, ISSIASoccerOffsetSyncDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render temporally shifted ISSIA camera pair.")
    parser.add_argument("--root", type=Path, default=ISSIA_SOCCER_ROOT, help="ISSIA-Soccer root path.")
    parser.add_argument("--cameras", type=int, nargs=2, default=[1, 2], help="Two camera IDs, shown top/bottom.")
    parser.add_argument("--start-frame", type=int, default=500, help="Base frame to start from.")
    parser.add_argument("--num-frames", type=int, default=120, help="Number of rendered frames.")
    parser.add_argument("--frame-step", type=int, default=1, help="Step in base-frame time.")
    parser.add_argument("--max-offset", type=int, default=60, help="Random max absolute frame offset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible offsets.")
    parser.add_argument("--width", type=int, default=960, help="Output width for each camera panel.")
    parser.add_argument("--fps", type=float, default=25.0, help="Output video FPS.")
    parser.add_argument("--show-boxes", action="store_true", help="Draw ISSIA annotation boxes.")
    parser.add_argument("--show", action="store_true", help="Open an OpenCV preview window while writing.")
    parser.add_argument("--output", type=Path, default=None, help="Output mp4 path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    end_frame = args.start_frame + (args.num_frames - 1) * args.frame_step
    dataset = ISSIASoccerOffsetSyncDataset(
        root,
        cameras=tuple(args.cameras),
        include_empty=True,
        load_images=True,
        return_tensors=False,
        image_mode="bgr",
        start_frame=args.start_frame,
        end_frame=end_frame,
        frame_step=args.frame_step,
        max_frame_offset=args.max_offset,
        random_seed=args.seed,
        reference_camera=args.cameras[0],
        force_reference_zero=True,
        allow_zero_non_reference=False,
    )

    if len(dataset) == 0:
        raise RuntimeError("Offset dataset is empty. Try a later start-frame or smaller max-offset.")

    offset_gt = dataset[0]["time_offset_gt"]
    print("Offset GT:")
    print(offset_gt)

    output_path = args.output
    if output_path is None:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cams = "_".join(str(cam) for cam in args.cameras)
        offsets = "_".join(
            f"cam{cam}{offset_gt['applied_frame_offsets'][cam]:+d}"
            for cam in args.cameras
        )
        output_path = DEBUG_OUTPUT_DIR / f"desync_{cams}_{offsets}_seed{args.seed}.mp4"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    first_canvas = build_canvas(dataset[0], args.cameras, args.width, args.show_boxes)
    height, width = first_canvas.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for idx in range(min(args.num_frames, len(dataset))):
            sample = dataset[idx]
            canvas = build_canvas(sample, args.cameras, args.width, args.show_boxes)
            writer.write(canvas)
            if args.show:
                cv2.imshow("ISSIA temporal offset debug", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
        dataset.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Wrote video: {output_path}")


def build_canvas(
    sample: Dict[str, Any],
    cameras: Tuple[int, int],
    output_width: int,
    show_boxes: bool,
) -> np.ndarray:
    panels = []
    for camera_id in cameras:
        camera_sample = sample["cameras"][camera_id]
        frame = camera_sample["image"].copy()
        if show_boxes:
            draw_boxes(frame, camera_sample["target"])
        draw_panel_text(frame, sample, camera_sample)
        panels.append(resize_to_width(frame, output_width))

    top, bottom = panels
    if top.shape[1] != bottom.shape[1]:
        bottom = cv2.resize(bottom, (top.shape[1], bottom.shape[0]))
    return np.vstack([top, bottom])


def draw_boxes(frame: np.ndarray, target: Dict[str, Any]) -> None:
    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        is_ball = int(label) == BALL_LABEL
        color = (0, 0, 255) if is_ball else (0, 255, 0)
        name = "ball" if is_ball else "person"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_panel_text(frame: np.ndarray, sample: Dict[str, Any], camera_sample: Dict[str, Any]) -> None:
    meta = camera_sample["meta"]
    gt = sample["time_offset_gt"]
    camera_id = meta["camera_id"]
    offset = gt["applied_frame_offsets"][camera_id]
    correction = gt["correction_frame_offsets"][camera_id]
    text_lines = [
        f"cam {camera_id}",
        f"base frame: {sample['base_frame_index']}",
        f"observed frame: {meta['observed_frame_index']}",
        f"applied offset: {offset:+d} frames",
        f"correction to sync: {correction:+d} frames",
    ]
    draw_text_block(frame, text_lines, origin=(20, 30))


def draw_text_block(frame: np.ndarray, lines: Sequence[str], origin: Tuple[int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    line_height = 28
    box_width = 430
    box_height = line_height * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 24), (x + box_width, y - 24 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_height), font, scale, (255, 255, 255), thickness)


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    height, current_width = frame.shape[:2]
    if current_width == width:
        return frame
    scale = width / float(current_width)
    return cv2.resize(frame, (width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
