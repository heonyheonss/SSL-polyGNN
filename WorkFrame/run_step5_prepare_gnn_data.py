# WorkFrame/run_step5_prepare_gnn_data.py
import pandas as pd
import os
import torch
from sklearn.preprocessing import StandardScaler # 추가
import pickle # 추가

# gnn_utils.py가 myutility 폴더에 있다고 가정
import sys

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'myutility')))
from myutility import gnn_utils as gu


def main():
    """5단계: GNN 학습을 위한 그래프 데이터셋을 생성하고 저장합니다."""
    print("--- 5단계 시작: GNN 데이터셋 준비 ---")

    # --- 1. 데이터 로드 ---
    imputed_data_path = 'E:/SSL-polyGNN/data/results/imputed_final_data.csv'
    try:
        df = pd.read_csv(imputed_data_path)
    except FileNotFoundError:
        print(f"오류: '{imputed_data_path}' 파일을 찾을 수 없습니다.")
        print("이전 단계(run_step4_impute_values.py)를 먼저 실행해주세요.")
        return

    # 물성 이름 목록 (SMILES 제외)
    properties = [col for col in df.columns if col not in ['SMILES', 'id']]
    print(f"대상 물성: {properties}")

    scaler = StandardScaler()
    df[properties] = scaler.fit_transform(df[properties])
    print("물성 데이터를 정규화했습니다.")

    # --- 2. 데이터셋 객체 생성 ---
    print("SMILES를 그래프 데이터로 변환 중... (시간이 걸릴 수 있습니다)")
    dataset = gu.MoleculeDataset(df, target_columns=properties)
    print(f"그래프 변환 완료. 총 {len(dataset)}개의 유효한 분자 데이터 생성.")

    # --- 3. 처리된 데이터 저장 ---
    processed_dir = 'E:/SSL-polyGNN/data/processed/'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    torch.save(dataset, os.path.join(processed_dir, 'gnn_dataset.pt'))
    with open(os.path.join(processed_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"처리된 데이터셋과 스케일러가 '{processed_dir}'에 저장되었습니다.")

    output_path = os.path.join(processed_dir, 'gnn_dataset.pt')
    # torch.save(dataset, output_path)

    print(f"\n--- 5단계 완료 ---")
    print(f"처리된 데이터셋이 '{output_path}'에 저장되었습니다.")


if __name__ == '__main__':
    main()