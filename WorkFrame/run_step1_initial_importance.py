# run_step1_initial_importance.py
import pandas as pd
import os
from myutility import descriptor_selector as ds
import yaml

# 결과를 저장할 디렉토리 생성
if not os.path.exists('results'):
    os.makedirs('results')



def main():
    with open("E:/SSL-polyGNN/configs/default.yaml", 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
        train_path = config["data"]["train_path"]
        properties = [prop for prop in config["data"]["properties"]]

    """1단계: 초기 피처 중요도를 계산하고 저장합니다."""
    print("--- 1단계 시작: 초기 피처 중요도 계산 ---")

    # 데이터 로드
    df_data = pd.read_csv(train_path)
    smiles = df_data['SMILES']

    # 모든 기술 지표 계산 (결과를 저장해두고 재사용 가능)
    df_descriptors = ds.calculate_descriptors(smiles)
    df_descriptors.to_pickle('results/all_descriptors.pkl')

    # 각 물성별 중요도 계산 및 저장
    for prop in properties:
        print(f"\n'{prop}' 물성의 피처 중요도를 계산합니다...")
        y = df_data[prop]
        importances = ds.get_feature_importances(df_descriptors, y, model_type='xgb')

        # 결과를 pickle 파일로 저장
        importances.to_pickle(f'results/01_initial_importance_{prop}.pkl')
        print(f"'{prop}'의 초기 중요도가 저장되었습니다.")

    print("\n--- 1단계 완료 ---")
    print("Jupyter Notebook에서 'results/01_initial_importance_Property_A.pkl' 등을 불러와")
    print("`importance.head(20).plot(kind='barh', title='Property_A Top 20 Descriptors')` 와 같이 시각화해보세요.")


if __name__ == '__main__':
    main()