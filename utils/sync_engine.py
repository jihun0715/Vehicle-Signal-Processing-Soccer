"""독립된 비전 노드 간의 객체 매칭 및 칼만 상태 벡터 NCC 기반 오프셋 추정 엔진.

작성자: Signal-sync 파트 신동하
"""

import numpy as np

class TrackMatcherSynchronizer:
    def __init__(self, max_lag_frames: int = 60, min_overlap_len: int = 40):
        """
        Args:
            max_lag_frames: 탐색할 최대 프레임 타임 딜레이 범위 (+/-)
            min_overlap_len: NCC 신뢰성을 확보하기 위해 필요한 최소 공통 프레임 길이
        """
        self.max_lag = max_lag_frames
        self.min_overlap = min_overlap_len

    def _calculate_vector_ncc(self, traj_A: np.ndarray, traj_B: np.ndarray):
        """[핵심 로직] 두 다차원 속도 시계열 벡터 간의 최고 NCC 점수와 최적의 Lag을 계산합니다."""
        lags = np.arange(-self.max_lag, self.max_lag + 1)
        best_score = -1.0
        best_lag = 0
        
        for lag in lags:
            # 시간축(Lag)으로 슬라이딩하며 두 신호가 겹치는 유효 구간 추출
            if lag < 0:
                valid_A = traj_A[-lag:]
                valid_B = traj_B[:lag]
            elif lag > 0:
                valid_A = traj_A[:-lag]
                valid_B = traj_B[lag:]
            else:
                valid_A = traj_A
                valid_B = traj_B

            length = min(len(valid_A), len(valid_B))
            if length < self.min_overlap:
                continue
                
            v_A = valid_A[:length]
            v_B = valid_B[:length]

            # 🛡️ Failsafe 예외 처리: 선수가 가만히 서 있는 정적 구간(노이즈)은 연산 스킵
            if np.std(v_A) < 0.05 or np.std(v_B) < 0.05:
                continue

            # 1. 공간 벡터 내적 (vx끼리, vy끼리 곱해서 합산 -> 차원 압축)
            dot_products = np.sum(v_A * v_B, axis=1)
            
            # 2. 정규화 분모 (에너지 노름) 계산 (스케일 차이 보정)
            norm_A = np.linalg.norm(v_A)
            norm_B = np.linalg.norm(v_B)
            
            if norm_A == 0 or norm_B == 0:
                score = 0.0
            else:
                # 3. 최종 NCC 점수 산출 (-1.0 ~ 1.0)
                score = float(np.sum(dot_products) / (norm_A * norm_B))
            
            # 피크(최고 점수) 갱신
            if score > best_score:
                best_score = score
                best_lag = int(lag)
                
        return best_score, best_lag

    def match_and_estimate(self, cam1_tracks: list, cam2_tracks: list):
        """
        [적용 로직] 팀원들의 WorldKalmanTracker 내부에 누적된 tracks 리스트를 인풋으로 받아,
        동일 인물 매칭(Correspondence)과 오프셋 추정을 동시에 수행합니다.
        """
        best_global_score = -1.0
        estimated_offset = 0
        matched_pair = None

        print("\n🔄 [Signal-sync] 칼만 상태 벡터 NCC 및 오프셋 추정 시작...")

        # 전수조사 (Brute-force Track-Pair Matching)
        for t1 in cam1_tracks:
            if not hasattr(t1, 'state_history') or len(t1.state_history) < self.min_overlap:
                continue
            
            # 🎯 [핵심] 칼만 필터 상태 벡터 x = [px, py, vx, vy] 중 인덱스 2, 3인 속도(vx, vy)만 슬라이싱!
            vel_A = np.array(t1.state_history)[:, 2:4] 

            for t2 in cam2_tracks:
                if not hasattr(t2, 'state_history') or len(t2.state_history) < self.min_overlap:
                    continue
                
                # 🎯 Camera 2의 속도(vx, vy) 성분 슬라이싱
                vel_B = np.array(t2.state_history)[:, 2:4]

                # 두 트랙 쌍에 대해 벡터 NCC를 돌려 가장 파형이 일치하는 오프셋(Lag) 도출
                score, lag = self._calculate_vector_ncc(vel_A, vel_B)

                # 가장 점수가 높은 트랙 쌍을 동일 인물로 간주하고 최종 오프셋 확정
                if score > best_global_score:
                    best_global_score = score
                    estimated_offset = lag
                    matched_pair = (t1.track_id, t2.track_id)

        if matched_pair is not None:
            print("🎯 [동기화 엔진 분석 성공]")
            print(f"  - 동일 인물 매칭 (Correspondence): Cam1_ID({matched_pair[0]}) <---> Cam2_ID({matched_pair[1]})")
            print(f"  - 파형 다차원 유사도 (Max NCC): {best_global_score:.4f}")
            print(f"  - 최종 추정 시간 위상차 (Offset): {estimated_offset} 프레임")
            estimated_offset = -estimated_offset
        else:
            print("⚠️ [경고] 비교 가능한 공통 궤적 신호가 부족합니다.")
            estimated_offset = 0

        return estimated_offset, matched_pair