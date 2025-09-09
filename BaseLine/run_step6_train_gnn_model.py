# BaseLine/run_step6_train_gnn_model.py
import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader # '.data'에서 '.loader'로 변경
# gnn_utils.py가 myutility 폴더에 있다고 가정
import sys

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'myutility')))
from myutility import gnn_utils as gu


def main():
    """6단계: GNN 모델을 K-fold 교차 검증으로 학습하고 최적 모델을 저장합니다."""
    print("--- 6단계 시작: GNN 모델 학습 ---")

    # --- 하이퍼파라미터 설정 ---
    N_SPLITS = 5
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # --- 1. 데이터 로드 ---
    dataset_path = 'E:/SSL-polyGNN/data/processed/gnn_dataset.pt'
    try:
        dataset = torch.load(dataset_path, weights_only=False, map_location=device)
    except FileNotFoundError:
        print(f"오류: '{dataset_path}' 파일을 찾을 수 없습니다.")
        print("5단계(run_step5_prepare_gnn_data.py)를 먼저 실행해주세요.")
        return
    except Exception as e:
        # 혹시 모를 다른 오류를 확인하기 위해 추가
        print(f"데이터 로딩 중 예상치 못한 오류 발생: {e}")
        # 오류의 상세 내용을 보기 위해 추가
        import traceback
        traceback.print_exc()
        return

    # 모델 입출력 크기 결정
    num_node_features = dataset[0].num_node_features
    num_targets = dataset[0].y.size(0)

    # --- 2. K-Fold 교차 검증 설정 ---
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Fold {fold + 1}/{N_SPLITS} =====")

        # --- 2-1. 데이터 분할 및 로더 생성 ---
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # --- 2-2. 모델, 옵티마이저, 손실 함수 초기화 ---
        model = gu.GCNNet(
            in_channels=num_node_features,
            hidden_channels=128,
            out_channels=num_targets
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.MSELoss()  # 평균 제곱 오차 (회귀 문제)

        best_val_loss = float('inf')

        # --- 2-3. 모델 학습 ---
        for epoch in range(1, EPOCHS + 1):
            train_loss = gu.train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = gu.eval_epoch(model, val_loader, criterion, device)

            if (epoch % 10 == 0) or (epoch == 1):
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 가장 좋은 성능의 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # 모델 저장 경로 생성
                model_dir = 'E:/SSL-polyGNN/model/'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_fold_{fold + 1}.pth'))

        print(f"Fold {fold + 1} 최적 Val Loss: {best_val_loss:.4f}")
        fold_histories.append(best_val_loss)

    # --- 3. 최종 성능 분석 ---
    avg_loss = np.mean(fold_histories)
    std_loss = np.std(fold_histories)
    print(f"\n--- 6단계 완료 ---")
    print(f"K-Fold 교차 검증 결과 (MSE Loss):")
    print(f"  평균: {avg_loss:.4f}")
    print(f"  표준편차: {std_loss:.4f}")
    print(f"최적 모델들이 'model/' 디렉토리에 저장되었습니다.")


if __name__ == '__main__':
    main()