# run_step4_impute_values.py
import pandas as pd
import os
from myutility import imputation_utils as iu
import yaml


def main():
    """4단계: 선택된 기술 지표를 사용하여 결측치를 예측하고 최종 데이터를 저장합니다."""
    print("--- 4단계 시작: 머신러닝 기반 결측치 예측 ---")

    # 결과 저장 디렉토리 생성
    output_dir = 'data/results/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. 필요 데이터 로드 ---
    # 원본 데이터 (결측치 포함)
    try:
        with open("E:/SSL-polyGNN/configs/default.yaml", 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
            train_path = config["data"]["train_path"]

        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print("에러: 'cleaned_polymer_data.csv' 파일을 찾을 수 없습니다.")
        return

    # 이전에 계산해 둔 전체 기술 지표
    df_descriptors = pd.read_pickle('results/all_descriptors.pkl')
    # 이전 단계에서 최종 선택된 기술 지표 목록
    df_selected_features = pd.read_csv('final_selected_descriptors.csv')

    df_imputed = df_original.copy()
    properties_to_impute = df_selected_features.columns

    # --- 2. 각 물성별로 결측치 예측 수행 ---
    for prop in properties_to_impute:
        print(f"\n'{prop}' 물성의 결측치 처리를 시작합니다.")

        # 해당 물성에 결측치가 있는지 확인
        if df_imputed[prop].isnull().sum() == 0:
            print(f"'{prop}' 물성에는 결측치가 없습니다. 다음으로 넘어갑니다.")
            continue

        # --- 2-1. 데이터 준비 ---
        # 이번 물성 예측에 사용할 기술 지표 이름 목록
        features_for_this_prop = df_selected_features[prop].tolist()

        # 훈련용(값이 있는) 데이터와 예측용(값이 없는) 데이터의 인덱스 분리
        train_indices = df_imputed[prop].notna()
        predict_indices = df_imputed[prop].isna()

        # 훈련용 데이터 (X, y)
        X_train = df_descriptors.loc[train_indices, features_for_this_prop]
        y_train = df_imputed.loc[train_indices, prop]

        # 예측용 데이터 (X)
        X_predict = df_descriptors.loc[predict_indices, features_for_this_prop]

        print(f"훈련 데이터: {len(X_train)}개, 예측할 데이터: {len(X_predict)}개")

        # --- 2-2. 모델 훈련 및 예측 ---
        # 모델 훈련
        model = iu.train_imputation_model(X_train, y_train, model_type='xgb')

        # 결측치 예측
        predicted_values = iu.predict_missing_values(model, X_predict)

        # --- 2-3. 예측값 채워넣기 ---
        df_imputed.loc[predict_indices, prop] = predicted_values
        print(f"'{prop}' 물성의 결측치 {len(predicted_values)}개를 성공적으로 채웠습니다.")

    # --- 3. 최종 결과 저장 ---
    output_path = os.path.join(output_dir, 'imputed_final_data.csv')
    df_imputed.to_csv(output_path, index=False)

    print("\n--- 4단계 완료 ---")
    print(f"결측치가 모두 채워진 최종 데이터가 '{output_path}'에 저장되었습니다.")
    # Jupyter Notebook에서 최종 결과 확인
    # display(df_imputed)


if __name__ == '__main__':
    main()