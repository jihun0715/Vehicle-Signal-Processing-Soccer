"""world 좌표계에서 선수 위치를 추적하는 Kalman filter tracker 모듈.

주요 클래스:
- `KalmanTrackerConfig`: gating, process noise, measurement noise, position covariance 하한,
  miss/hit 기준 등 tracker 파라미터를 담는다.
- `WorldObservation`: projection이 만든 한 개의 world-frame 측정값 `(x, y)`와 bbox metadata를 담는다.
- `WorldTrack`: track id, state, covariance, hit/miss, reliability를 가진 mutable track 상태다.
- `WorldKalmanTracker`: predict/update, Mahalanobis gating, Hungarian assignment, track 생성/삭제를 수행한다.

상태 벡터는 PLAN.md에 정리한 INHA `brain_data.cpp` 스타일을 따른다:

    x = [px, py, vx, vy]

여기서 px/py는 image 좌표가 아니라 공통 world 좌표계상의 위치다.
process noise 설정값 `q_pos`, `q_vel`은 표준편차로 해석하고, Q에는 제곱한
variance를 dt 스케일과 함께 적용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - exercised only when SciPy is missing.
    linear_sum_assignment = None


@dataclass(frozen=True)
class KalmanTrackerConfig:
    gate_chi2: float = 9.21
    big_cost: float = 1e9
    mahalanobis_weight: float = 1.0
    l2_weight: float = 0.0
    l2_norm_m: float = 1.0
    q_pos: float = 0.02
    q_vel: float = 0.50
    brake_accel: float = 0.8
    min_speed_eps: float = 0.03
    max_predict_dt_sec: float = 0.20
    init_pos_std: float = 2.0
    init_vel_std: float = 1.0
    measurement_std: float = 0.25
    min_position_std: float = 0.0
    reliability_sigma_ref: float = 0.60
    max_misses: int = 30
    min_hits: int = 1

    @classmethod
    def from_config(cls) -> "KalmanTrackerConfig":
        import config as project_config

        return cls(
            gate_chi2=project_config.TRACK_GATE_CHI2,
            big_cost=project_config.TRACK_BIG_COST,
            mahalanobis_weight=project_config.TRACK_COST_MAHALANOBIS_WEIGHT,
            l2_weight=project_config.TRACK_COST_L2_WEIGHT,
            l2_norm_m=project_config.TRACK_COST_L2_NORM_M,
            q_pos=project_config.TRACK_Q_POS,
            q_vel=project_config.TRACK_Q_VEL,
            brake_accel=project_config.TRACK_BRAKE_ACCEL,
            min_speed_eps=project_config.TRACK_MIN_SPEED_EPS,
            max_predict_dt_sec=project_config.TRACK_MAX_PREDICT_DT_SEC,
            init_pos_std=project_config.TRACK_INIT_POS_STD,
            init_vel_std=project_config.TRACK_INIT_VEL_STD,
            measurement_std=project_config.TRACK_MEASUREMENT_STD,
            min_position_std=getattr(project_config, "TRACK_MIN_POSITION_STD", 0.0),
            reliability_sigma_ref=project_config.TRACK_RELIABILITY_SIGMA_REF,
            max_misses=project_config.TRACK_MAX_MISSES,
            min_hits=project_config.TRACK_MIN_HITS,
        )


@dataclass(frozen=True)
class WorldObservation:
    """One projected detection in the common world coordinate system."""

    x: float
    y: float
    timestamp_sec: Optional[float] = None
    confidence: float = 1.0
    camera_id: Optional[int] = None
    frame_index: Optional[int] = None
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    detection_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "timestamp_sec": self.timestamp_sec,
            "confidence": float(self.confidence),
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "bbox_xyxy": self.bbox_xyxy,
            "detection_id": self.detection_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class WorldTrack:
    """Mutable Kalman track state in world coordinates."""

    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    hits: int
    misses: int
    age: int
    last_timestamp_sec: float
    last_update_timestamp_sec: float
    reliability: float = 0.0
    last_cost: Optional[float] = None
    confidence: float = 1.0
    camera_id: Optional[int] = None
    frame_index: Optional[int] = None
    last_observation: Optional[WorldObservation] = None

    @property
    def position(self) -> Tuple[float, float]:
        return (float(self.state[0]), float(self.state[1]))

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self.state[2]), float(self.state[3]))

    def is_confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "position": self.position,
            "velocity": self.velocity,
            "state": [float(value) for value in self.state],
            "covariance": self.covariance.tolist(),
            "hits": self.hits,
            "misses": self.misses,
            "age": self.age,
            "last_timestamp_sec": self.last_timestamp_sec,
            "last_update_timestamp_sec": self.last_update_timestamp_sec,
            "reliability": float(self.reliability),
            "last_cost": self.last_cost,
            "confidence": float(self.confidence),
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
        }


class WorldKalmanTracker:
    """Multi-object Kalman tracker over world-frame point observations."""

    def __init__(
        self,
        tracker_config: Optional[KalmanTrackerConfig] = None,
        *,
        camera_id: Optional[int] = None,
    ) -> None:
        self.config = tracker_config or KalmanTrackerConfig()
        self.camera_id = camera_id
        self._tracks: List[WorldTrack] = []
        self._next_track_id = 1
        self._last_timestamp_sec: Optional[float] = None

    @classmethod
    def from_config(cls, *, camera_id: Optional[int] = None) -> "WorldKalmanTracker":
        return cls(KalmanTrackerConfig.from_config(), camera_id=camera_id)

    @property
    def tracks(self) -> List[WorldTrack]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks = []
        self._next_track_id = 1
        self._last_timestamp_sec = None

    def predict(self, timestamp_sec: float) -> List[WorldTrack]:
        timestamp = float(timestamp_sec)
        for track in self._tracks:
            track.age += 1
            self._predict_track(track, timestamp)
        self._last_timestamp_sec = timestamp
        return self.tracks

    def update(
        self,
        observations: Iterable[WorldObservation],
        *,
        timestamp_sec: Optional[float] = None,
    ) -> List[WorldTrack]:
        obs_list = list(observations)
        timestamp = self._resolve_timestamp(obs_list, timestamp_sec)

        for track in self._tracks:
            track.age += 1
            self._predict_track(track, timestamp)

        matched_track_indices = set()
        matched_obs_indices = set()

        if self._tracks and obs_list:
            cost_matrix = self._build_cost_matrix(self._tracks, obs_list)
            for track_index, obs_index in self._assign(cost_matrix):
                cost = float(cost_matrix[track_index, obs_index])
                if cost >= self.config.big_cost:
                    continue
                self._update_track(self._tracks[track_index], obs_list[obs_index], timestamp, cost)
                matched_track_indices.add(track_index)
                matched_obs_indices.add(obs_index)

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_track_indices:
                track.misses += 1
                track.last_cost = None
                track.reliability = self._compute_reliability(track.covariance)

        for obs_index, obs in enumerate(obs_list):
            if obs_index not in matched_obs_indices:
                self._tracks.append(self._initialize_track(obs, timestamp))

        self._tracks = [track for track in self._tracks if track.misses <= self.config.max_misses]
        self._last_timestamp_sec = timestamp
        return self.tracks

    def get_tracks(self, *, confirmed_only: bool = False) -> List[WorldTrack]:
        if not confirmed_only:
            return self.tracks
        return [track for track in self._tracks if track.is_confirmed(self.config.min_hits)]

    def _resolve_timestamp(
        self,
        observations: Sequence[WorldObservation],
        timestamp_sec: Optional[float],
    ) -> float:
        if timestamp_sec is not None:
            return float(timestamp_sec)
        for obs in observations:
            if obs.timestamp_sec is not None:
                return float(obs.timestamp_sec)
        if self._last_timestamp_sec is not None:
            return float(self._last_timestamp_sec)
        return 0.0

    def _initialize_track(self, obs: WorldObservation, timestamp: float) -> WorldTrack:
        state = np.array([obs.x, obs.y, 0.0, 0.0], dtype=np.float64)
        covariance = np.diag(
            [
                self.config.init_pos_std ** 2,
                self.config.init_pos_std ** 2,
                self.config.init_vel_std ** 2,
                self.config.init_vel_std ** 2,
            ]
        ).astype(np.float64)
        covariance = _apply_position_covariance_floor(covariance, self.config.min_position_std)
        track = WorldTrack(
            track_id=self._next_track_id,
            state=state,
            covariance=covariance,
            hits=1,
            misses=0,
            age=1,
            last_timestamp_sec=float(timestamp),
            last_update_timestamp_sec=float(timestamp),
            confidence=float(obs.confidence),
            camera_id=obs.camera_id if obs.camera_id is not None else self.camera_id,
            frame_index=obs.frame_index,
            last_observation=obs,
        )
        track.reliability = self._compute_reliability(track.covariance)
        self._next_track_id += 1
        return track

    def _predict_track(self, track: WorldTrack, timestamp: float) -> None:
        raw_dt = float(timestamp) - float(track.last_timestamp_sec)
        if raw_dt < 0.0:
            raw_dt = 0.0
        dt = min(raw_dt, self.config.max_predict_dt_sec)
        if dt <= 0.0:
            track.last_timestamp_sec = float(timestamp)
            return

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        control = self._brake_control(track.state[2:4], dt)
        process_noise = self._process_noise(dt)

        track.state = transition.dot(track.state) + control
        track.covariance = transition.dot(track.covariance).dot(transition.T) + process_noise
        track.covariance = _apply_position_covariance_floor(
            track.covariance,
            self.config.min_position_std,
        )
        track.last_timestamp_sec = float(timestamp)
        track.reliability = self._compute_reliability(track.covariance)

    def _brake_control(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        if speed <= self.config.min_speed_eps or self.config.brake_accel <= 0.0:
            return np.zeros(4, dtype=np.float64)

        accel_mag = min(self.config.brake_accel, speed / max(dt, 1e-9))
        accel = -accel_mag * velocity / speed
        return np.array(
            [
                0.5 * dt * dt * accel[0],
                0.5 * dt * dt * accel[1],
                dt * accel[0],
                dt * accel[1],
            ],
            dtype=np.float64,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        q_pos_var = float(self.config.q_pos) ** 2
        q_acc_var = float(self.config.q_vel) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        noise = np.zeros((4, 4), dtype=np.float64)
        noise[0, 0] = q_acc_var * 0.25 * dt4 + q_pos_var * dt
        noise[1, 1] = q_acc_var * 0.25 * dt4 + q_pos_var * dt
        noise[0, 2] = q_acc_var * 0.5 * dt3
        noise[2, 0] = noise[0, 2]
        noise[1, 3] = q_acc_var * 0.5 * dt3
        noise[3, 1] = noise[1, 3]
        noise[2, 2] = q_acc_var * dt2
        noise[3, 3] = q_acc_var * dt2
        return noise

    def _build_cost_matrix(
        self,
        tracks: Sequence[WorldTrack],
        observations: Sequence[WorldObservation],
    ) -> np.ndarray:
        costs = np.full((len(tracks), len(observations)), self.config.big_cost, dtype=np.float64)
        for track_index, track in enumerate(tracks):
            for obs_index, obs in enumerate(observations):
                cost = self._association_cost(track, obs)
                costs[track_index, obs_index] = cost
        return costs

    def _association_cost(self, track: WorldTrack, obs: WorldObservation) -> float:
        measurement = obs.position
        residual = measurement - track.state[:2]
        measurement_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        measurement_noise = np.eye(2, dtype=np.float64) * (self.config.measurement_std ** 2)
        innovation_cov = (
            measurement_matrix.dot(track.covariance).dot(measurement_matrix.T) + measurement_noise
        )
        mahalanobis_d2 = _quadratic_form(innovation_cov, residual)
        if mahalanobis_d2 > self.config.gate_chi2:
            return self.config.big_cost

        l2 = float(np.linalg.norm(residual))
        l2_norm = max(float(self.config.l2_norm_m), 1e-9)
        return (
            self.config.mahalanobis_weight * mahalanobis_d2
            + self.config.l2_weight * (l2 / l2_norm)
        )

    def _assign(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        if cost_matrix.size == 0:
            return []

        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost_matrix)
            return [(int(row), int(col)) for row, col in zip(rows, cols)]

        candidates = []
        for row in range(cost_matrix.shape[0]):
            for col in range(cost_matrix.shape[1]):
                candidates.append((float(cost_matrix[row, col]), row, col))
        candidates.sort(key=lambda item: item[0])

        used_rows = set()
        used_cols = set()
        matches = []
        for cost, row, col in candidates:
            if cost >= self.config.big_cost:
                break
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            matches.append((row, col))
        return matches

    def _update_track(
        self,
        track: WorldTrack,
        obs: WorldObservation,
        timestamp: float,
        cost: float,
    ) -> None:
        measurement_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        measurement_noise = np.eye(2, dtype=np.float64) * (self.config.measurement_std ** 2)
        measurement = obs.position
        residual = measurement - measurement_matrix.dot(track.state)
        innovation_cov = (
            measurement_matrix.dot(track.covariance).dot(measurement_matrix.T) + measurement_noise
        )

        kalman_gain = _kalman_gain(track.covariance, measurement_matrix, innovation_cov)
        identity = np.eye(4, dtype=np.float64)
        innovation_matrix = identity - kalman_gain.dot(measurement_matrix)

        track.state = track.state + kalman_gain.dot(residual)
        track.covariance = (
            innovation_matrix.dot(track.covariance).dot(innovation_matrix.T)
            + kalman_gain.dot(measurement_noise).dot(kalman_gain.T)
        )
        track.covariance = _apply_position_covariance_floor(
            track.covariance,
            self.config.min_position_std,
        )
        track.hits += 1
        track.misses = 0
        track.last_timestamp_sec = float(timestamp)
        track.last_update_timestamp_sec = float(timestamp)
        track.last_cost = float(cost)
        track.confidence = float(obs.confidence)
        track.camera_id = obs.camera_id if obs.camera_id is not None else self.camera_id
        track.frame_index = obs.frame_index
        track.last_observation = obs
        track.reliability = self._compute_reliability(track.covariance)

    def _compute_reliability(self, covariance: np.ndarray) -> float:
        pos_cov = covariance[:2, :2]
        det = max(float(np.linalg.det(pos_cov)), 0.0)
        sigma_geo = float(np.sqrt(np.sqrt(det)))
        sigma_ref = max(float(self.config.reliability_sigma_ref), 1e-9)
        return 1.0 / (1.0 + sigma_geo / sigma_ref)


def _quadratic_form(matrix: np.ndarray, vector: np.ndarray) -> float:
    try:
        solved = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(matrix).dot(vector)
    return float(vector.T.dot(solved))


def _kalman_gain(
    covariance: np.ndarray,
    measurement_matrix: np.ndarray,
    innovation_cov: np.ndarray,
) -> np.ndarray:
    cross_cov = covariance.dot(measurement_matrix.T)
    try:
        return np.linalg.solve(innovation_cov.T, cross_cov.T).T
    except np.linalg.LinAlgError:
        return cross_cov.dot(np.linalg.pinv(innovation_cov))


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _apply_position_covariance_floor(
    covariance: np.ndarray,
    min_position_std: float,
) -> np.ndarray:
    covariance = _symmetrize(np.asarray(covariance, dtype=np.float64))
    min_std = float(min_position_std)
    if min_std <= 0.0:
        return covariance

    min_var = min_std * min_std
    pos_cov = _symmetrize(covariance[:2, :2])
    eigvals, eigvecs = np.linalg.eigh(pos_cov)
    clamped_eigvals = np.maximum(eigvals, min_var)
    covariance[:2, :2] = eigvecs.dot(np.diag(clamped_eigvals)).dot(eigvecs.T)
    return _symmetrize(covariance)
