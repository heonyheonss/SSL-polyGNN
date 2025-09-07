import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import yaml

# 1. RDKit descriptor generation function
def generate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # calculate all descriptors in RDKit
    descriptors = {name: float(func(mol)) for name, func in Descriptors.descList}
    return descriptors

# 2. Extract importance for target from DataFrames
def extract_importance_for(df_full, df_descriptors, target_property, is_top_only=True):
    df_selection = df_full.dropna(subset=[target_property])
    X = df_selection[df_descriptors.columns]
    y = df_selection[target_property]

    # --- [최종 해결책 적용] ---
    # 1. 모든 열을 강제로 숫자 타입으로 변환 (오류 발생 시 NaN으로)
    X = X.apply(pd.to_numeric, errors='coerce')
    # 2. NaN 값을 0으로 채우기
    X = X.fillna(0)
    # 3. float32 범위를 벗어나는 모든 값을 최대/최소값으로 강제 조정 (Clipping)
    f32_info = np.finfo(np.float32)
    X = X.clip(lower=f32_info.min, upper=f32_info.max)
    # --- [해결책 끝] ---

    # RF training and extract feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # make importance dataframe
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
    top_n = 20
    top_features = feature_importance.nlargest(top_n)

    print(f"===== '{target_property}' 예측을 위한 상위 {top_n}개 디스크립터 =====")
    print(top_features)

    if not is_top_only:
        return feature_importance
    else:
        return top_features

def main():
    with open("E:/SSL-polyGNN/configs/default.yaml", 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
        train_path = config["data"]["train_path"]
        target_properties = [prop for prop in config["data"]["properties"]]

    df_final = pd.read_csv(train_path)

    descriptor_data = df_final['SMILES'].apply(generate_descriptors)
    df_descriptors = pd.DataFrame(list(descriptor_data))

    # 원본 데이터와 디스크립터 데이터 합치기
    df_full = pd.concat([df_final, df_descriptors], axis=1)

    for target in target_properties:
        top_features = extract_importance_for(df_full, df_descriptors, target)
        top_n = len(top_features)

        # 피처 중요도 시각화
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_features, y=top_features.index)
        plt.title(f'Top {top_n} Feature Importances for {target} Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Descriptor')
        plt.tight_layout()

        # 그래프 저장
        output_filename = f'feature_importance_{target}.png'
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"\n피처 중요도 그래프가 '{output_filename}'으로 저장되었습니다.")