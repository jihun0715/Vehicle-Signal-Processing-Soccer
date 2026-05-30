# Vehicle-Signal-Processing-Soccer 계획

## 목표

여러 축구 경기 카메라 사이의 시간 오프셋을 추정한다. 이를 위해 선수의 이동 궤적과 포즈 신호를 비교하며, 전체 흐름은 객체 검출, 월드 좌표계 투영, 트래킹, 카메라 간 트랙 매칭, 포즈 추정, 포즈 기반 시간 보정으로 구성한다.

첫 구현 목표는 학습보다 추론 중심 파이프라인이다. YOLO와 MMPose는 프로젝트 전용 파인튜닝 없이 pretrained model을 사용한다.

## 현재 데이터 계층

`data/` 아래에 구현된 내용:

- `ISSIASoccerFrameDataset`: 카메라 프레임을 하나씩 반환한다.
- `ISSIASoccerSyncDataset`: 동일한 frame index의 여러 카메라 샘플을 묶어서 반환한다.
- `create_issia_dataloader`: variable-length target을 처리하는 collate 함수와 함께 PyTorch `DataLoader`를 생성한다.
- `read_issia_annotations`: ISSIA `.xgtf` annotation을 읽어서 person/ball bbox로 변환한다.

샘플 형식:

- `image`: RGB 또는 BGR 프레임. 옵션에 따라 Torch tensor로 변환된다.
- `target["boxes"]`: image-space xyxy bbox.
- `target["labels"]`: 내부 class ID. `0=person`, `1=ball`.
- `target["object_ids"]`: ISSIA VIPER annotation의 object ID.
- `meta`: camera ID, frame index, timestamp, video path, image size.

이 dataloader는 detector 평가뿐 아니라 이후 projection, tracking, pose, temporal calibration 디버깅 입력으로 사용한다.

## 파이프라인

1. 객체 검출

   Ultralytics YOLO x-size 계열 weight를 사용해서 player와 ball을 검출한다. `YOLO26x` 또는 다른 x-size checkpoint로 쉽게 바꿀 수 있도록 detector wrapper는 model path 기반으로 설계한다.

   초기 출력:

   - camera ID
   - frame index와 timestamp
   - image 좌표계 bbox
   - class label
   - confidence

2. 공통 월드 좌표계로 투영

   image-space detection을 축구장 기준의 공통 pitch/world 좌표계로 변환한다.

   초기 목표:

   - 사람 bbox의 bottom-center 또는 foot point를 field XY로 투영
   - 가능한 경우 ball center를 field XY로 투영
   - MMPose crop에 사용할 원본 camera-space bbox metadata 유지

3. 카메라별 트래킹

   투영된 detection을 이용해 Kalman tracker를 구현한다. 주요 참고 구현은 `INHA-Soccer/INHA-Player/src/brain/src/brain_data.cpp`이다.

   유지하고 싶은 설계:

   - state vector `x = [px, py, vx, vy]`
   - constant-velocity transition
   - 비현실적인 velocity drift를 줄이기 위한 optional deceleration control
   - 시간 간격을 반영한 process noise
   - field `x, y` observation 기반 measurement update
   - chi-square threshold 기반 Mahalanobis gating
   - Hungarian matching을 통한 global one-to-one assignment
   - 수치 안정성을 위한 Joseph-form covariance update
   - position covariance 기반 reliability score

4. 카메라 간 트랙 매칭

   각 카메라에서 안정적인 tracklet이 만들어진 뒤, 월드 좌표계 궤적을 비교해 서로 다른 카메라의 트랙을 매칭한다.

   후보 feature:

   - 겹치는 시간 구간
   - field XY trajectory distance
   - velocity direction agreement
   - track reliability와 detection confidence
   - coarse offset hypothesis를 위한 temporal-shifted trajectory similarity

5. 포즈 추정

   MMPose는 파인튜닝 없이 사용한다.

   우선 추천하는 방향:

   - 매칭된 player bbox crop에 대해 top-down 2D pose estimation 수행
   - robust front end로 RTMPose 계열 2D body keypoint 사용
   - 2D keypoint sequence를 MotionBERT 또는 VideoPose3D 같은 MMPose human3d pose-lifter로 3D pose로 lifting

   이유:

   - 이 프로젝트는 시간 window를 이미 사용하므로 temporal pose lifter와 잘 맞는다.
   - monocular 3D pose의 절대 scale은 불안정할 수 있지만, normalized motion과 joint direction 신호는 NCC에 사용할 수 있다.
   - 3D pose가 broadcast-like soccer frame에서 불안정하면 2D pose feature를 fallback으로 사용할 수 있다.

6. 포즈 벡터화

   각 track의 pose output을 time-series vector로 변환한다.

   후보 vector:

   - root-relative joint coordinates
   - normalized limb direction vectors
   - joint velocities
   - gait-like periodic motion을 위한 lower-body joints
   - occluded joint 영향을 줄이기 위한 confidence-weighted features

7. Temporal calibration

   매칭된 track의 pose vector window에 대해 normalized cross-correlation을 수행해 camera time offset을 추정한다.

   출력:

   - camera pair `(A, B)`
   - frames 단위와 seconds 단위의 estimated offset
   - NCC peak score
   - offset 추정에 사용된 matched track IDs

## 마일스톤

1. 데이터 sanity check

   `ISSIASoccerFrameDataset`과 `ISSIASoccerSyncDataset`으로 ISSIA video와 annotation을 정상 로드한다.

2. Detector wrapper

   dataloader sample을 입력받아 detection record를 저장하는 YOLO inference module을 추가한다.

3. Projection module

   camera calibration input을 추가하고 detection foot point를 field coordinate로 변환한다.

4. Tracker module

   INHA-style Kalman tracker를 Python으로 port한다. YOLO detection에 붙이기 전에 synthetic trajectory test를 먼저 작성한다.

5. Track matching module

   world-frame trajectory similarity를 이용해 카메라별 track을 매칭한다.

6. Pose module

   MMPose 2D pose inference를 추가하고, 이후 3D pose-lifting backend를 붙인다.

7. Temporal calibration module

   matched track pose window에 대해 NCC 기반 offset estimation을 구현한다.

8. End-to-end script

   ISSIA의 선택된 camera pair를 detection, tracking, matching, pose, offset estimation까지 한 번에 실행한다.

## 현재 리스크

- ISSIA annotation frame index가 일부 video frame count보다 크게 들어있는 경우가 있어, dataloader는 읽을 수 없는 frame을 건너뛴다.
- camera calibration file이 아직 repo에 없다.
- pretrained YOLO는 작은 ball을 놓칠 수 있다. Temporal calibration의 1차 목표는 player detection이다.
- MMPose 기반 monocular 3D pose는 metric scale이 안정적이지 않을 수 있으므로 NCC에는 normalized pose feature를 우선 사용한다.
- 카메라 간 track matching 품질은 projection 정확도와 track continuity에 크게 의존한다.
