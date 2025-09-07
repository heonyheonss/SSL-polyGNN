# WorkFrame/run_step7_create_submission.py
import pandas as pd
import torch
import os
from tqdm import tqdm
import pickle

# myutility 폴더의 gnn_utils.py를 사용하기 위한 경로 설정
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'myutility')))
import gnn_utils as gu


def main():
    """7단계: 학습된 모델들로 테스트 데이터의 물성을 예측하고 submission.csv를 생성합니다."""
    print("--- 7단계 시작: 최종 제출 파일 생성 ---")

    # --- 1. 설정 및 필요 데이터/모델 로드 ---

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # 모델이 저장된 경로와 데이터 경로
    model_dir = 'E:/SSL-polyGNN/model/'
    data_dir = 'E:/SSL-polyGNN/data/'  # test.csv, sample_submission.csv가 있는 경로

    # 테스트 데이터 로드
    try:
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        sample_submission_df = pd.read_csv(os.path.join(data_dir, 'submission/sample_submission.csv'))
    except FileNotFoundError as e:
        print(f"오류: {e}. 'data' 폴더에 test.csv와 sample_submission.csv가 있는지 확인해주세요.")
        return

    # K-Fold로 학습된 모델 5개 불러오기
    models = []
    num_folds = 5
    # 모델의 입출력 크기 설정 (이전 단계와 동일해야 함)
    # 원자 특징 36개, 예측할 물성 5개
    in_channels, out_channels = 36, 5

    for i in range(1, num_folds + 1):
        model_path = os.path.join(model_dir, f'best_model_fold_{i}.pth')
        if not os.path.exists(model_path):
            print(f"오류: {model_path}를 찾을 수 없습니다. 6단계 모델 학습을 먼저 완료해주세요.")
            return

        model = gu.GCNNet(in_channels=in_channels, hidden_channels=128, out_channels=out_channels)
        # map_location을 사용하여 어떤 환경에서든 모델을 안전하게 불러옴
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # 모델을 '평가 모드'로 설정 (Dropout 등 비활성화)
        models.append(model)

    print(f"성공적으로 {len(models)}개의 모델을 불러왔습니다.")

    scaler_path = 'E:/SSL-polyGNN/data/processed/scaler.pkl'
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"오류: {scaler_path}를 찾을 수 없습니다. 5단계를 다시 실행하여 스케일러를 생성해주세요.")
        return

    # --- 2. 앙상블 예측 수행 ---

    all_predictions = []

    # tqdm을 사용하여 예측 진행 상황 표시
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        smiles = row['SMILES']

        # SMILES를 그래프 데이터로 변환
        # 예측 시에는 y값이 필요 없으므로 임의의 값(torch.zeros)을 넣어줌
        graph_data = gu.smiles_to_graph(smiles, torch.zeros(out_channels))
        if graph_data is None:
            print("failed to generate graph")
            # 변환 실패 시, 0으로 채운 예측값 사용 (혹은 평균값 등 다른 전략도 가능)
            all_predictions.append([0] * out_channels)
            continue

        graph_data = graph_data.to(device)

        fold_predictions = []
        with torch.no_grad():  # 예측 시에는 gradient 계산이 필요 없음
            for model in models:
                # DataLoader 없이 단일 데이터를 예측할 때는 batch 정보가 없으므로 직접 추가
                # PyG 최신 버전에서는 data.batch를 자동으로 처리해주기도 함
                if 'batch' not in graph_data:
                    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long).to(device)

                pred = model(graph_data)
                fold_predictions.append(pred.cpu().numpy().flatten())

        # 5개 모델 예측 결과의 평균을 계산 (앙상블)
        ensemble_prediction = sum(fold_predictions) / len(fold_predictions)
        all_predictions.append(ensemble_prediction)

    # --- 3. 제출 파일 생성 및 저장 ---

    # 예측 결과를 데이터프레임으로 변환
    pred_df = pd.DataFrame(all_predictions, columns=sample_submission_df.columns[1:])

    pred_df.iloc[:, :] = scaler.inverse_transform(pred_df)
    print("예측값을 원래 스케일로 복원했습니다.")

    # id 컬럼과 예측 결과 데이터프레임을 합치기
    submission_df = pd.concat([test_df[['id']], pred_df], axis=1)

    # 최종 제출 파일 저장
    submission_path = 'E:/SSL-polyGNN/data/submission/submission.csv'
    submission_df.to_csv(submission_path, index=False)

    print("\n--- 7단계 완료 ---")
    print(f"최종 제출 파일이 '{submission_path}'에 성공적으로 저장되었습니다.")
    print("Kaggle에 제출하여 성능을 확인해보세요!")


if __name__ == '__main__':
    main()