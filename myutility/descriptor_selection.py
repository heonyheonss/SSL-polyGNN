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
    descriptors = {name: func(mol) for name, func in Descriptors.descList}
    return descriptors

# 2. Extract importance for target from DataFrames
def extract_importance_for(df_full, df_descriptors, target_property, is_top_only=True):
    df_selection = df_full.dropna(subset=[target_property])

    # X (descriptor results), y (target property) separation
    X = df_selection[df_descriptors.columns]
    y = df_selection[target_property]

    # Inf, NaN inplace with 0 (resulted from descriptor calculation)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

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

# 3. filtering descriptors by their correlation coefficient
def filtering_descriptors(df_descriptors, initial_features):
    # generate descriptor Dataframe with initial features
    df_selected = df_descriptors[initial_features]

    # calculate correlation matrix
    corr_matrix = df_selected.corr().abs()

    # generate set of drop features
    to_drop = set()
    # remove features with high correlation coefficient
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # corr coeff >= 0.8, not listed in "to_drop"
            if (corr_matrix.iloc[i, j] >= 0.8) and (corr_matrix.columns[j] not in to_drop) and (
                    corr_matrix.columns[i] not in to_drop):

                # calculate mean correlation of each features with all the other features
                mean_corr_i = corr_matrix.iloc[i, :].mean()
                mean_corr_j = corr_matrix.iloc[j, :].mean()

                # add higher one to_drop list
                if mean_corr_i > mean_corr_j:
                    to_drop.add(corr_matrix.columns[i])
                    print(
                        f"상관관계 쌍 발견: ('{corr_matrix.columns[i]}', '{corr_matrix.columns[j]}'), 상관계수: {corr_matrix.iloc[i, j]:.2f}")
                    print(f" -> '{corr_matrix.columns[i]}' 제거 (평균 상관계수 더 높음: {mean_corr_i:.2f} > {mean_corr_j:.2f})\n")
                else:
                    to_drop.add(corr_matrix.columns[j])
                    print(
                        f"상관관계 쌍 발견: ('{corr_matrix.columns[i]}', '{corr_matrix.columns[j]}'), 상관계수: {corr_matrix.iloc[i, j]:.2f}")
                    print(f" -> '{corr_matrix.columns[j]}' 제거 (평균 상관계수 더 높음: {mean_corr_j:.2f} > {mean_corr_i:.2f})\n")

    # final features list
    final_features = [f for f in initial_features if f not in to_drop]

    print("=" * 50)
    print("제거될 디스크립터:")
    print(list(to_drop))
    print("\n최종 선택된 디스크립터:")
    print(final_features)
    print(f"\n총 {len(initial_features)}개에서 {len(to_drop)}개가 제거되어 {len(final_features)}개가 선택되었습니다.")

    return final_features

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

# TODO - make new python file which uses code beneath here and returns final features
def get_desc_for_using():
    with open("E:/SSL-polyGNN/configs/default.yaml", 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
        train_path = config["data"]["train_path"]
        target_properties = [prop for prop in config["data"]["properties"]]

    df_final = pd.read_csv(train_path)

    descriptor_data = df_final['SMILES'].apply(generate_descriptors)
    df_descriptors = pd.DataFrame(list(descriptor_data))

    df_full = pd.concat([df_final, df_descriptors], axis=1)

    for target in target_properties:
        top_features = extract_importance_for(df_full, df_descriptors, target)
        final_features = filtering_descriptors(df_descriptors, top_features)

    return final_features