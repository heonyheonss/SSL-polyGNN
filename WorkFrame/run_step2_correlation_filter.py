# run_step2_correlation_filter.py
import pandas as pd
import os
from myutility import descriptor_selector as ds


def main():
    """2단계: 상관관계가 높은 피처를 제거하고 중요도를 다시 저장합니다."""
    print("\n--- 2단계 시작: 상관관계 기반 필터링 ---")

    df_descriptors = pd.read_pickle('results/all_descriptors.pkl')
    properties = [f.split('_')[-1].replace('.pkl', '') for f in os.listdir('results') if
                  f.startswith('01_initial_importance')]

    for prop in properties:
        print(f"\n'{prop}' 물성의 상관관계를 필터링합니다...")
        initial_importance = pd.read_pickle(f'results/01_initial_importance_{prop}.pkl')

        # 상관관계 필터링 실행
        kept_features = ds.filter_by_correlation(df_descriptors, initial_importance, threshold=0.8)

        # 살아남은 피처들의 중요도만 필터링하여 저장
        final_importance = initial_importance.loc[kept_features]
        final_importance.to_pickle(f'results/02_final_importance_{prop}.pkl')
        print(f"'{prop}'의 상관관계 필터링 후 중요도가 저장되었습니다.")

    print("\n--- 2단계 완료 ---")
    print("이제 각 물성별로 상관관계가 높은 기술 지표들이 정리되었습니다.")


if __name__ == '__main__':
    main()