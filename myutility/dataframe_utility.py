import pandas as pd
from myutility import featuring

# TODO - input: target column, output: X,Y for fitting

def get_structure_target_set(df, target):
    df_clean = df[['SMILES', target]].dropna().copy()
    return df_clean

def get_feature(df):
    feature_data = df['SMILES'].progress_apply(featuring.smiles_to_features)
    feature_df = pd.DataFrame(feature_data.tolist(), index=df.index)
    final_df = pd.concat([df, feature_df], axis=1)
    final_df = final_df.dropna()
    return final_df

def multi_target_dataset():
    return