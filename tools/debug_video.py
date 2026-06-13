"""카메라 간 temporal offset dataset을 영상으로 확인하는 CLI tool.

클래스는 없고, 주요 함수는 다음과 같다.
- `main`: offset dataloader를 만들고 GT offset을 출력한 뒤 debug mp4를 저장한다.
- `build_canvas`: 여러 카메라 프레임을 상하분할 canvas로 합친다.
- `draw_boxes`, `draw_panel_text`: annotation bbox와 frame/offset 정보를 영상 위에 그린다.

실행 예:
    python -m tools.debug_video
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from config import (
    DEBUG_OUTPUT_DIR,
    DEBUG_VIDEO_BATCH_SIZE,
    DEBUG_VIDEO_CAMERAS,
    DEBUG_VIDEO_FPS,
    DEBUG_VIDEO_FRAME_STEP,
    DEBUG_VIDEO_NUM_FRAMES,
    DEBUG_VIDEO_NUM_WORKERS,
    DEBUG_VIDEO_SHOW_BOXES,
    DEBUG_VIDEO_START_FRAME,
    DEBUG_VIDEO_WIDTH,
    ISSIA_SOCCER_ROOT,
    OFFSET_ALLOW_ZERO_NON_REFERENCE,
    OFFSET_FORCE_REFERENCE_ZERO,
    OFFSET_MAX_FRAME_OFFSET,
    OFFSET_RANDOM_SEED,
    OFFSET_REFERENCE_CAMERA,
)
from data import BALL_LABEL, create_issia_offset_dataloader
from utils import TrackMatcherSynchronizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render temporally shifted ISSIA camera pair. Defaults come from config.py."
    )
    parser.add_argument("--show", action="store_true", help="Open an OpenCV preview window while writing.")
    parser.add_argument("--show-boxes", action="store_true", default=None, help="Draw ISSIA annotation boxes.")
    parser.add_argument("--output", type=Path, default=None, help="Output mp4 path.")
    return parser.parse_args()


def load_settings(cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        root=Path(ISSIA_SOCCER_ROOT),
        cameras=tuple(DEBUG_VIDEO_CAMERAS),
        start_frame=DEBUG_VIDEO_START_FRAME,
        num_frames=DEBUG_VIDEO_NUM_FRAMES,
        frame_step=DEBUG_VIDEO_FRAME_STEP,
        max_offset=OFFSET_MAX_FRAME_OFFSET,
        seed=OFFSET_RANDOM_SEED,
        reference_camera=OFFSET_REFERENCE_CAMERA,
        force_reference_zero=OFFSET_FORCE_REFERENCE_ZERO,
        allow_zero_non_reference=OFFSET_ALLOW_ZERO_NON_REFERENCE,
        width=DEBUG_VIDEO_WIDTH,
        fps=DEBUG_VIDEO_FPS,
        batch_size=DEBUG_VIDEO_BATCH_SIZE,
        num_workers=DEBUG_VIDEO_NUM_WORKERS,
        show_boxes=DEBUG_VIDEO_SHOW_BOXES if cli_args.show_boxes is None else cli_args.show_boxes,
        show=cli_args.show,
        output=cli_args.output,
    )

class MockTrackInstance:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.state_history = [] # [[px, py, vx, vy], ...] 구조 축적용 버퍼

def main() -> None:
    args = load_settings(parse_args())
    args.root = args.root.expanduser()
    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")
    if args.show:
        print(f"OpenCV window preview enabled. DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    else:
        print("OpenCV window preview disabled. Use --show to open a live window.")

    end_frame = args.start_frame + (args.num_frames - 1) * args.frame_step
    loader = create_issia_offset_dataloader(
        args.root,
        cameras=tuple(args.cameras),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_empty=True,
        load_images=True,
        return_tensors=False,
        image_mode="bgr",
        start_frame=args.start_frame,
        end_frame=end_frame,
        frame_step=args.frame_step,
        max_frame_offset=args.max_offset,
        random_seed=args.seed,
        reference_camera=args.reference_camera,
        force_reference_zero=args.force_reference_zero,
        allow_zero_non_reference=args.allow_zero_non_reference,
    )
    dataset = loader.dataset

    if len(dataset) == 0:
        raise RuntimeError("Offset dataset is empty. Try a later start-frame or smaller max-offset.")

    offset_gt = dataset[0]["time_offset_gt"]
    print("Offset GT:")
    print(offset_gt)
    print(
        f"Rendering {min(args.num_frames, len(dataset))} frames "
        f"with batch_size={args.batch_size}, num_workers={args.num_workers}"
    )

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
    print(f"Writing video to: {output_path}")

    # (선수 한 명의 대표 궤적을 전수 빌드업하여 NCC 딜레이 탐색 테스트를 진행하기 위함)
    camera_tracks_buffer = {int(cam): MockTrackInstance(track_id=int(cam * 10)) for cam in args.cameras}

    try:
        written = 0
        should_stop = False
        for batch in loader:
            for sample in batch["samples"]:
                if written >= args.num_frames:
                    should_stop = True
                    break

                # 매 프레임 스트리밍되는 샘플에서 선수들의 기하학적 2D 변위 및 속도 성분 누적 추출
                for camera_id in args.cameras:
                    cam_sample = sample["cameras"][camera_id]
                    boxes = cam_sample["target"]["boxes"]
                    
                    # 프레임 내에 선수 객체가 존재한다면 (가장 첫 번째 대표 선수의 움직임을 궤적 신호로 샘플링)
                    if len(boxes) > 0:
                        box = boxes[0]
                        px = (box[0] + box[2]) / 2.0
                        py = box[3] # Foot point
                        
                        # 이전 프레임 위치 기반 실시간 변위 분산 속도 역산 변환 (Pseudo Kalman Filter State)
                        history = camera_tracks_buffer[camera_id].state_history
                        if len(history) > 0:
                            vx = px - history[-1][0]
                            vy = py - history[-1][1]
                        else:
                            vx, vy = 0.0, 0.0
                        
                        # 팀 아키텍처 규격 상태 벡터 x = [px, py, vx, vy] 형태로 기록 적재
                        history.append([px, py, vx, vy])
                    else:
                        # 결측치 방어 로직 (선수가 화면에서 일시 소실된 정적 구간 보정)
                        history = camera_tracks_buffer[camera_id].state_history
                        if len(history) > 0:
                            history.append([history[-1][0], history[-1][1], 0.0, 0.0])
                        else:
                            history.append([0.0, 0.0, 0.0, 0.0])

                canvas = build_canvas(sample, args.cameras, args.width, args.show_boxes)
                writer.write(canvas)
                written += 1
                if written == 1 or written % 50 == 0:
                    print(f"Rendered {written} frames")
                if args.show:
                    cv2.imshow("ISSIA temporal offset debug", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        should_stop = True
                        break
            if should_stop:
                break
            print(f"\n🎬 비디오 스트리밍 렌더링 완료. Signal-sync 엔진 연동 및 시간 정합 연산 개시...")
        
        # 1. 동기화 클래스 인스턴스화
        synchronizer = TrackMatcherSynchronizer(max_lag_frames=60, min_overlap_len=40)
        
        # 2. 데이터로더 내부 딕셔너리에 숨겨진 '강제로 넣은 정답 오프셋 지표(Ground Truth)' 파악
        # (Cam 1번 기준 대비 다른 카메라가 몇 프레임 밀렸는지 기록된 컬렉션 활용)
        applied_gt = offset_gt["applied_frame_offsets"]
        correction_gt = offset_gt["correction_frame_offsets"]
        
        # 3. 비디오 채널 리스트 추출
        cam_ids = sorted(list(camera_tracks_buffer.keys()))
        ref_cam = cam_ids[0]
        tgt_cam = cam_ids[1]
        
        print("\n" + "="*60)
        print("📊 [Signal-sync 파트최종 통합 정량 평가 리포트 (A+ Metric)]")
        print("-"*60)
        
        # 핵심 다차원 벡터 NCC 엔진 메서드 슛팅 구동!
        # 두 카메라 트랙 버퍼 인스턴스 리스트를 토스합니다.
        tracks_A = [camera_tracks_buffer[ref_cam]]
        tracks_B = [camera_tracks_buffer[tgt_cam]]
        
        estimated_delay, matched_info = synchronizer.match_and_estimate(tracks_A, tracks_B)
        
        # 대현님이 주입한 실제 비전 노드 간 상대 위상차 정답 수식 도출
        # correction_frame_offsets은 동기화를 위해 적용해야 할 보정값이므로, 실제 지연 정답은 부호가 반대입니다.
        true_relative_delay = correction_gt[tgt_cam] - correction_gt[ref_cam]
        
        # 4. 정량적 오차 분석 산출
        absolute_error = abs(true_relative_delay - estimated_delay)
        rmse_score = np.sqrt(absolute_error ** 2) # 단일 조합 검증이므로 절대 오차가 곧 RMSE에 대응
        
        print("-"*60)
        print(f"  🎥 비전 노드 정합 분석 대조 (Cam {ref_cam} <---> Cam {tgt_cam}):")
        print(f"    - 주입된 딜레이 참값 (Ground Truth)   : {true_relative_delay:+d} frames")
        print(f"    - NCC 신호처리 추정치 (Estimated)     : {estimated_delay:+d} frames")
        print(f"    - 시스템 최종 정량적 오차 (RMSE)       : {rmse_score:.4f} frames")
        print("="*60 + "\n")
    finally:
        writer.release()
        if hasattr(dataset, "close"):
            dataset.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Wrote video: {output_path}")


def build_canvas(
    sample: Dict[str, Any],
    cameras: Sequence[int],
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
