# imputation_utils.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def train_imputation_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'xgb'
):
    """
    결측치 예측을 위한 모델을 훈련시킵니다.

    Args:
        X_train (pd.DataFrame): 훈련용 피처 데이터 (기술 지표).
        y_train (pd.Series): 훈련용 타겟 데이터 (물성 값).
        model_type (str): 사용할 모델 ('xgb' 또는 'rf').

    Returns:
        훈련된 머신러닝 모델 객체.
    """
    if model_type == 'xgb':
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError("model_type은 'xgb' 또는 'rf'여야 합니다.")

    model.fit(X_train, y_train)
    return model


def predict_missing_values(model, X_to_predict: pd.DataFrame) -> list:
    """
    훈련된 모델을 사용하여 결측치를 예측합니다.

    Args:
        model: 훈련된 머신러닝 모델 객체.
        X_to_predict (pd.DataFrame): 예측을 수행할 피처 데이터.

    Returns:
        list: 예측된 값들의 리스트.
    """
    predictions = model.predict(X_to_predict)
    return predictions