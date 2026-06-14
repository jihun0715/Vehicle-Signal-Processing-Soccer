"""동기화 알고리즘 결과 시각화 및 발표용 차트 생성 모듈.

작성자: Signal-sync 파트 신동하
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_sync_results(
    ncc_scores: np.ndarray, 
    lags: np.arange, 
    best_lag: float, 
    traj_A: np.ndarray, 
    traj_B: np.ndarray, 
    save_dir: str = "debug_outputs"
):
    """
    발표 자료에 즉시 삽입할 수 있는 2종의 고해상도 품질 그래프를 생성합니다.
    
    Args:
        ncc_scores: 각 lag별 계산된 NCC 점수 배열
        lags: 탐색 범위 프레임 배열 (-max_lag ~ max_lag)
        best_lag: 알고리즘이 최종 추정한 오프셋 값 (소수점 포함 가능)
        traj_A: 기준 카메라(Cam 1)의 매칭된 선수 속도 벡터 시계열 (N, 2)
        traj_B: 대상 카메라(Cam 2)의 매칭된 선수 속도 벡터 시계열 (M, 2)
    """
    import os
    from pathlib import Path
    
    # 저장 경로 생성
    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1차원 속도 성분 크기(Magnitude) 추출 (시각화용 대조를 위함)
    speed_A = np.linalg.norm(traj_A, axis=1)
    speed_B = np.linalg.norm(traj_B, axis=1)

    # -----------------------------------------------------------------
    # 그래프 1: NCC 글로벌 스코어 곡선 및 추정 피크 지점
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 4.5))
    plt.plot(-lags, ncc_scores, color='#1f77b4', linestyle='-', linewidth=2, label='Global NCC Curve')
    plt.axvline(x=best_lag, color='#d62728', linestyle='--', linewidth=2, 
                label=f'Estimated Peak Offset: {best_lag:.2f} frames')
    plt.axvline(x=72.0, color='#2ca02c', linestyle=':', linewidth=2, label='Ground Truth: 72.00 frames')
    
    plt.title('Temporal Synchronization: Global NCC Peak Search', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Time Lag (Frames)', fontsize=11)
    plt.ylabel('Normalized Cross-Correlation Score', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=10)
    
    chart1_path = out_path / "sync_ncc_peak_curve.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved presentation chart (NCC Peak): {chart1_path}")

    # -----------------------------------------------------------------
    # 그래프 2: 동기화 전(Before) vs 동기화 후(After) 속도 파형 정합 대조
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharey=True)
    
    # 2-1. Before Alignment (정합 전)
    ax1.plot(speed_A, color='#1f77b4', linewidth=1.8, label='Camera 1 (Reference)')
    ax1.plot(speed_B, color='#ff7f0e', linewidth=1.5, alpha=0.7, label='Camera 2 (Time-Delayed)')
    ax1.set_title('Velocity Waveform Profile [Before Temporal Alignment]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speed Magnitude (m/s)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    # 2-2. After Alignment (추정 오프셋 적용 정합 후)
    ax2.plot(speed_A, color='#1f77b4', linewidth=1.8, label='Camera 1 (Reference)')
    
    # 추정된 lag 프레임만큼 시간축 평행 이동 보정
    aligned_time_axis = np.arange(len(speed_B)) + best_lag
    ax2.plot(aligned_time_axis, speed_B, color='#d62728', linewidth=1.5, alpha=0.8, label='Camera 2 (Phase-Aligned)')
    
    ax2.set_title('Velocity Waveform Profile [After Temporal Alignment]', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Global Frame Index', fontsize=11)
    ax2.set_ylabel('Speed Magnitude (m/s)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    chart2_path = out_path / "sync_waveform_alignment_comparison.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved presentation chart (Before/After): {chart2_path}")