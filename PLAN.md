# Vehicle-Signal-Processing-Soccer Plan

## Goal

Estimate temporal offsets between soccer cameras by comparing player motion trajectories and pose signals after detection, projection, tracking, cross-camera matching, and pose lifting.

The first implementation target is an inference-first pipeline. YOLO and MMPose are used as pretrained models without project-specific fine-tuning.

## Current Data Layer

Implemented under `data/`:

- `ISSIASoccerFrameDataset`: returns one camera frame at a time.
- `ISSIASoccerSyncDataset`: returns the same frame index across multiple cameras.
- `create_issia_dataloader`: builds a PyTorch `DataLoader` with a variable-target collate function.
- `read_issia_annotations`: parses ISSIA `.xgtf` annotations into person and ball boxes.

Sample format:

- `image`: RGB or BGR frame, optionally converted to a Torch tensor.
- `target["boxes"]`: xyxy image-space boxes.
- `target["labels"]`: internal class IDs, `0=person`, `1=ball`.
- `target["object_ids"]`: VIPER object IDs from ISSIA annotations.
- `meta`: camera ID, frame index, timestamp, video path, image size.

The dataloader is intended to feed both detector evaluation and downstream pipeline debugging.

## Pipeline

1. Object detection

   Use Ultralytics YOLO x-size model weights for player and ball detection. Keep the detector wrapper model-path based so `YOLO26x` or another x-size checkpoint can be swapped without touching downstream code.

   Initial outputs:

   - camera ID
   - frame index and timestamp
   - detected bbox in image coordinates
   - class label
   - confidence

2. Projection to common world frame

   Convert image-space detections into a shared pitch/world coordinate system.

   Initial target:

   - person foot point or bbox bottom-center projected to field XY
   - ball center projected to field XY when possible
   - keep camera-space bbox as observation metadata for MMPose crops

3. Per-camera tracking

   Implement Kalman tracking after projection, using `INHA-Soccer/INHA-Player/src/brain/src/brain_data.cpp` as the main reference.

   Reference behavior to preserve:

   - state vector `x = [px, py, vx, vy]`
   - constant-velocity transition
   - optional deceleration control to prevent unrealistic velocity drift
   - time-normalized process noise
   - measurement update on field `x, y`
   - Mahalanobis gating with chi-square threshold
   - global one-to-one assignment with Hungarian matching
   - Joseph-form covariance update
   - reliability score derived from position covariance

4. Cross-camera track matching

   After each camera has stable tracks, match tracklets across cameras by comparing world-frame trajectories.

   Candidate features:

   - overlapping time interval
   - field XY trajectory distance
   - velocity direction agreement
   - track reliability and detection confidence
   - temporal-shifted trajectory similarity for coarse offset hypotheses

5. Pose estimation

   Use MMPose without fine-tuning.

   Recommended first path:

   - run 2D top-down pose estimation on matched player bbox crops
   - use RTMPose-style 2D body keypoints as the robust front end
   - lift 2D keypoint sequences with an MMPose human3d pose-lifter such as MotionBERT or VideoPose3D

   Reason:

   - the project already has time windows, so temporal pose lifters fit naturally
   - absolute monocular 3D scale may be unreliable, but normalized motion and joint-direction signals are still useful for NCC
   - 2D pose can also be retained as a fallback feature if 3D pose is unstable on broadcast soccer frames

6. Pose vectorization

   Convert pose output for each track into a time-series vector.

   Candidate vectors:

   - root-relative joint coordinates
   - normalized limb direction vectors
   - joint velocities
   - selected lower-body joints for gait-like periodic motion
   - confidence-weighted features to suppress occluded joints

7. Temporal calibration

   Estimate camera time offset with normalized cross-correlation over matched track pose vectors.

   Output:

   - camera pair `(A, B)`
   - estimated offset in frames and seconds
   - NCC peak score
   - supporting matched track IDs

## Milestones

1. Data sanity

   Load ISSIA videos and annotations through `ISSIASoccerFrameDataset` and `ISSIASoccerSyncDataset`.

2. Detector wrapper

   Add a YOLO inference module that accepts dataloader samples and writes detection records.

3. Projection module

   Add camera calibration input and convert detection foot points into field coordinates.

4. Tracker module

   Port the INHA-style Kalman tracker into Python with tests on synthetic trajectories before using YOLO detections.

5. Track matching module

   Match per-camera tracks using world-frame trajectory similarity.

6. Pose module

   Add MMPose 2D pose inference and then a 3D pose-lifting backend.

7. Temporal calibration module

   Build NCC-based offset estimation over matched track pose windows.

8. End-to-end script

   Run a selected camera pair from ISSIA through detection, tracking, matching, pose, and offset estimation.

## Open Risks

- ISSIA annotation frame indices extend beyond some video frame counts, so dataloaders currently skip unreadable frames.
- Camera calibration files are not part of the current repo yet.
- Pretrained YOLO may miss the small ball; player detection is the first priority for temporal calibration.
- Monocular 3D pose from MMPose may not be metrically stable in broadcast-like views; use normalized pose features for NCC.
- Cross-camera matching quality depends heavily on projection accuracy and track continuity.
