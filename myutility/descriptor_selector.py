# descriptor_selector.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# tqdm의 pandas integration 활성화
tqdm.pandas()


def calculate_descriptors(smiles_series: pd.Series) -> pd.DataFrame:
    """
    SMILES 시리즈로부터 RDKit 2D 기술 지표를 계산합니다.

    Args:
        smiles_series (pd.Series): 분자 구조의 SMILES 문자열을 담은 pandas 시리즈.

    Returns:
        pd.DataFrame: 계산된 기술 지표를 담은 데이터프레임.
                      계산 실패 시 해당 행은 NaN 값으로 채워집니다.
    """
    print("RDKit 기술 지표 계산을 시작합니다...")

    # 계산할 기술 지표 목록 생성
    desc_list = [desc[0] for desc in Descriptors._descList]

    results = []
    for smiles in tqdm(smiles_series, desc="Calculating Descriptors"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 분자 구조 생성 실패 시, NaN으로 채워진 딕셔너리 추가
            results.append({name: np.nan for name in desc_list})
            continue

        try:
            # 각 기술 지표 계산
            descriptors = {name: func(mol) for name, func in Descriptors._descList}
            results.append(descriptors)
        except Exception as e:
            print(f"SMILES '{smiles}' 처리 중 오류 발생: {e}")
            results.append({name: np.nan for name in desc_list})

    df_descriptors = pd.DataFrame(results)

    # 무한대 값을 NaN으로 변환
    df_descriptors.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 각 기술 지표(열)의 평균값으로 결측치(NaN) 채우기
    print("결측치를 각 기술 지표의 평균값으로 채웁니다.")
    df_descriptors.fillna(df_descriptors.mean(), inplace=True)

    # 그래도 남은 결측치가 있다면 0으로 채움 (평균 계산이 불가능한 경우 대비)
    df_descriptors.fillna(0, inplace=True)

    print(f"기술 지표 계산 완료! 최종 형태: {df_descriptors.shape}")
    return df_descriptors


def get_feature_importances(X: pd.DataFrame, y: pd.Series, model_type: str = 'xgb') -> pd.Series:
    """
    주어진 데이터로 모델을 훈련시키고 피처 중요도를 반환합니다.

    Args:
        X (pd.DataFrame): 입력 피처 (기술 지표).
        y (pd.Series): 타겟 변수 (물성).
        model_type (str): 사용할 모델 ('xgb' 또는 'rf').

    Returns:
        pd.Series: 피처 이름과 중요도를 담은 시리즈 (중요도 순으로 정렬).
    """
    if model_type == 'xgb':
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError("model_type은 'xgb' 또는 'rf'여야 합니다.")

    # --- [최종 해결책 적용] ---
    # 1. 모든 열을 강제로 숫자 타입으로 변환 (오류 발생 시 NaN으로)
    X = X.apply(pd.to_numeric, errors='coerce')
    # 2. NaN 값을 0으로 채우기
    X = X.fillna(0)
    # 3. float32 범위를 벗어나는 모든 값을 최대/최소값으로 강제 조정 (Clipping)
    f32_info = np.finfo(np.float32)
    X = X.clip(lower=f32_info.min, upper=f32_info.max)
    # --- [해결책 끝] ---

    # 타겟 변수에 결측치가 있다면 해당 샘플 제거
    valid_indices = y.dropna().index
    X_train = X.loc[valid_indices]
    y_train = y.loc[valid_indices]

    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False)


def filter_by_correlation(X_df: pd.DataFrame, importance_series: pd.Series, threshold: float = 0.8) -> list:
    """
    상관관계가 높은 피처 그룹에서 중요도가 가장 높은 피처만 남깁니다.

    Args:
        X_df (pd.DataFrame): 필터링할 피처 데이터프레임.
        importance_series (pd.Series): 각 피처의 중요도.
        threshold (float): 상관관계 임계값.

    Returns:
        list: 살아남은 피처의 이름 리스트.
    """
    # 현재 유효한 피처들만으로 상관관계 행렬 계산
    valid_features = [feat for feat in importance_series.index if feat in X_df.columns]
    corr_matrix = X_df[valid_features].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for column in upper.columns:
        # 임계값을 넘는 상관관계를 가진 피처들
        correlated_features = upper.index[upper[column] > threshold].tolist()
        if correlated_features:
            # 자기 자신과 상관관계 높은 피처들을 그룹으로 묶음
            group = [column] + correlated_features
            # 그룹 내에서 중요도가 가장 낮은 피처들을 제거 목록에 추가
            # 중요도 시리즈를 기준으로 정렬
            group_importances = importance_series.loc[group]
            # 가장 중요도가 높은 피처를 제외한 나머지를 제거 목록에 추가
            to_drop.update(group_importances.sort_values(ascending=False).index[1:])

    # 살아남을 피처 목록
    features_to_keep = [feat for feat in valid_features if feat not in to_drop]

    print(f"상관관계 필터링: {len(to_drop)}개의 기술 지표 제거. {len(features_to_keep)}개 남음.")
    return features_to_keep


def select_final_descriptors(
        prop_importance_dict: dict,
        cumulative_threshold: float = 0.7
) -> dict:
    """
    여러 물성에 대해 '동일한 개수'의 최종 기술 지표를 선택합니다.

    Args:
        prop_importance_dict (dict): {'물성1': 중요도1, '물성2': 중요도2, ...} 형태의 딕셔너리.
        cumulative_threshold (float): 최소 누적 중요도 합계.

    Returns:
        dict: {'물성1': [선택된 기술 지표], ...} 형태의 딕셔너리.
    """
    selected_descriptors = {}
    descriptor_counts = []

    # 1. 각 물성별로 누적 중요도 0.7을 만족하는 최소 기술 지표 집합 찾기
    for prop, importances in prop_importance_dict.items():
        cum_importance = importances.cumsum()
        # 누적 중요도가 임계값을 처음으로 넘는 지점의 인덱스 + 1 이 개수
        count = (cum_importance > cumulative_threshold).idxmax()
        num_to_select = importances.index.get_loc(count) + 1

        selected_descriptors[prop] = importances.head(num_to_select).index.tolist()
        descriptor_counts.append(len(selected_descriptors[prop]))

    # 2. 모든 물성에서 선택된 기술 지표 개수 중 '최소값'을 기준으로 통일
    min_count = min(descriptor_counts)
    print(f"모든 물성에 공통으로 적용될 기술 지표 개수: {min_count}")

    final_selection = {}
    for prop, descriptors in selected_descriptors.items():
        # 각 리스트를 min_count 만큼 자름 (이미 중요도 순으로 정렬되어 있음)
        final_selection[prop] = descriptors[:min_count]

    return final_selection


def save_results_to_csv(final_descriptor_dict: dict, filename: str):
    """
    최종 선택된 기술 지표 목록을 CSV 파일로 저장합니다.

    Args:
        final_descriptor_dict (dict): 최종 기술 지표 딕셔너리.
        filename (str): 저장할 파일 이름.
    """
    df_to_save = pd.DataFrame(final_descriptor_dict)
    df_to_save.to_csv(filename, index=False)
    print(f"최종 결과가 '{filename}' 파일로 저장되었습니다.")