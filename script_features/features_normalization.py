# features_normalization.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def normalize_all_versions(df_features, drop_cols=["label", "participant", "time_center", "start", "end"]):
    """
    Normalization with 'minmax', 'zscore', 'robust'
    """
    results = {}
    scalers = {
        "minmax": MinMaxScaler(),
        "zscore": StandardScaler(),
        "robust": RobustScaler()
    }
    
    df = df_features.copy()
    
    keep_cols = [c for c in drop_cols if c in df.columns]
    keep = df[keep_cols]
    features = df.drop(columns=keep_cols, errors='ignore')
    
    for name, scaler in scalers.items():
        scaled = scaler.fit_transform(features)
        df_scaled = pd.DataFrame(scaled, columns=features.columns)
        df_final = pd.concat([df_scaled, keep.reset_index(drop=True)], axis=1)
        results[name] = df_final
    
    return results
