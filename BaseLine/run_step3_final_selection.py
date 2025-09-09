# run_step3_final_selection.py
import pandas as pd
import os
from myutility import descriptor_selector as ds


def main():
    """3단계: 최종 기술 지표를 선택하고 CSV 파일로 저장합니다."""
    print("\n--- 3단계 시작: 최종 기술 지표 선정 및 저장 ---")

    properties = [f.split('_')[-1].replace('.pkl', '') for f in os.listdir('results') if
                  f.startswith('02_final_importance')]

    prop_importance_dict = {}
    for prop in properties:
        prop_importance_dict[prop] = pd.read_pickle(f'results/02_final_importance_{prop}.pkl')

    # 최종 기술 지표 선택
    final_selection = ds.select_final_descriptors(prop_importance_dict, cumulative_threshold=0.7)

    # 결과 저장
    ds.save_results_to_csv(final_selection, 'final_selected_descriptors.csv')

    print("\n--- 3단계 완료 ---")
    print("Jupyter Notebook에서 `pd.read_csv('final_selected_descriptors.csv')`로 최종 결과를 확인하세요.")


if __name__ == '__main__':
    main()