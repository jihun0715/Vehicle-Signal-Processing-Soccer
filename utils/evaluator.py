import numpy as np

class SyncEvaluator:
    """
    타임 오프셋 동기화 알고리즘의 정량적 성능을 평가하는 클래스입니다.
    여러 테스트 케이스(예: 다양한 카메라 쌍, 다양한 시드)의 결과를 누적하여 
    최종 RMSE, MAE 등을 계산합니다.
    """
    def __init__(self):
        self.gt_offsets = []
        self.estimated_offsets = []

    def add_result(self, gt: float, est: float):
        """단일 테스트(1개 비디오 쌍)의 결과를 누적합니다."""
        self.gt_offsets.append(float(gt))
        self.estimated_offsets.append(float(est))

    def calculate_metrics(self):
        """누적된 결과를 바탕으로 평가 지표를 계산하여 반환합니다."""
        if not self.gt_offsets:
            print("⚠️ [경고] 평가할 데이터가 없습니다.")
            return None

        gts = np.array(self.gt_offsets)
        ests = np.array(self.estimated_offsets)
        
        # 오차(Error) 배열
        errors = ests - gts

        # 1. RMSE (Root Mean Square Error): 큰 오차에 민감 (가장 중요)
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # 2. MAE (Mean Absolute Error): 전체적인 평균 오차 수준
        mae = np.mean(np.abs(errors))
        
        # 3. Standard Deviation of Error: 오차가 얼마나 들쭉날쭉한지(안정성)
        std_error = np.std(errors)
        
        # 4. Max Error: 가장 크게 빗나간 최악의 케이스
        max_error = np.max(np.abs(errors))

        return {
            "Total_Tests": len(gts),
            "RMSE": rmse,
            "MAE": mae,
            "Error_STD": std_error,
            "Max_Error": max_error
        }

    def print_summary(self):
        """발표용/디버깅용 결과 요약 텍스트를 터미널에 예쁘게 출력합니다."""
        metrics = self.calculate_metrics()
        if not metrics:
            return

        print("\n" + "="*45)
        print(" 📊 [Signal-sync] 최종 정량 평가 보고서")
        print("="*45)
        print(f" ▪ 테스트 횟수 (N)      : {metrics['Total_Tests']} 회")
        print(f" ▪ RMSE (평균 제곱근 오차): {metrics['RMSE']:.4f} 프레임")
        print(f" ▪ MAE (평균 절대 오차)  : {metrics['MAE']:.4f} 프레임")
        print(f" ▪ 오차 표준편차 (STD)   : {metrics['Error_STD']:.4f} 프레임")
        print(f" ▪ 최대 오차 (Worst)     : {metrics['Max_Error']:.4f} 프레임")
        print("="*45 + "\n")