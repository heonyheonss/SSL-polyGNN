import yaml
import pandas as pd
import numpy as np

from myutility import descriptor_selection
from myutility.descriptor_selection import generate_descriptors, extract_importance_for

# 3. filtering descriptors by their correlation coefficient
def filtering_descriptors(df_descriptors, initial_features):
    # generate descriptor Dataframe with initial features
    print(initial_features)
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

def get_desc_for_using(df_, target):

    descriptor_data = df_['SMILES'].apply(generate_descriptors)
    df_descriptors = pd.DataFrame(list(descriptor_data))

    df_full = pd.concat([df_, df_descriptors], axis=1)

    top_features = extract_importance_for(df_full, df_descriptors, target)
    final_features = filtering_descriptors(df_descriptors, top_features)

    return final_features.index.tolist()

def main():
    with open("E:/SSL-polyGNN/configs/default.yaml", 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
        train_path = config["data"]["train_path"]
        target_properties = [prop for prop in config["data"]["properties"]]

    df = pd.read_csv(train_path)
    # --- 데이터 로딩 및 준비 (기존 코드와 동일) ---
    # df_final, df_descriptors, target_properties 등은 이미 준비되었다고 가정합니다.
    # df_full = pd.concat([df_final, df_descriptors], axis=1)

    # 1. 결과를 저장할 빈 딕셔너리를 생성합니다.
    property_descriptors_dict = {}

    # 각 target property에 대해 루프를 실행합니다.
    for target in target_properties:
        print(f"Processing property: {target}")

        # 2. 딕셔너리에 결과를 저장합니다.
        # key는 property 이름, value는 디스크립터 이름 리스트입니다.
        property_descriptors_dict[target] = get_desc_for_using(df, target)

    # 3. 루프가 끝난 후, 딕셔너리를 사용해 최종 데이터프레임을 생성합니다.
    df_final_descriptors = pd.DataFrame(property_descriptors_dict)

    print("\n===== 최종 디스크립터 목록 =====")
    print(df_final_descriptors)

    return df_final_descriptors

# main() 함수를 실행하면 아래와 같은 형태의 DataFrame이 반환됩니다.