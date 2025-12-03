"""
Completed pipeline to process WESAD and to extract features 
"""

import os
import pandas as pd
import numpy as np

from load_signal import load_wesad_subject

# preprocessing
from process1 import (
    preprocess_breath_signals,
    preprocess_hr_signals,
    preprocess_eda_signals,
    preprocess_emg_signals,
)
# feature extraction + normalization + plotting
from features1 import (
    extract_all_features,
    normalize_all_versions,
    plot_peaks_by_section,
)

FS = 700  # (no resampling)


def process_wesad_subject(subject_id, wesad_path):
    """
    only one subject

    """
    # 1. Load data
    subject_path = os.path.join(wesad_path, subject_id)
    df = load_wesad_subject(subject_path)
    
    
    print(f"Respiration range: [{df['BF'].min():.3f}, {df['BF'].max():.3f}] %")
    print(f"ECG range: [{df['HR'].min():.3f}, {df['HR'].max():.3f}] mV")
    if 'EDA' in df.columns:
        print(f"EDA range: [{ df['EDA'].min():.3f}, {df['EDA'].max():.3f}] µS")
    if 'EMG' in df.columns:
        print(f"EMG range: {df['EMG'].min():.3f}, {df['EMG'].max():.3f}] mV")
    

    # 2. Preprocessing
    print("[STEP 1/4] PREPROCESSING SIGNALS")
   
    df_clean = preprocess_breath_signals(df, fs_custom=FS)
    df_clean = preprocess_hr_signals(df_clean)
    df_clean = preprocess_eda_signals(df_clean)
    df_clean = preprocess_emg_signals(df_clean)
      
    # Cleaned signals
    os.makedirs(f"output/{subject_id}", exist_ok=True)
    df_clean.to_csv(f"output/{subject_id}/clean_signals.csv", index=False)
    
    
    # 3. Peaks
    print("[STEP 2/4] PEAK DETECTION VISUALIZATION")
    
    try:
        print("Plotting peaks by label section")
        plot_peaks_by_section(df_clean, fs=FS)
    except Exception as e:
        print(f"[WARN] Plotting failed (skipping): {e}")
    
   
    # 4. Feature extraction
    print("[STEP 3/4] FEATURE EXTRACTION")
    features_df = extract_all_features(df_clean, fs=FS)
    features_df['participant'] = subject_id
    
    # features raw
    features_df.to_csv(f"output/{subject_id}/features_raw.csv", index=False)
    
    print(f"Features extracted")
    print(f"    Shape: {features_df.shape}")
    print(f"    Samples (windows): {len(features_df)}")
    # metadata: label, time_center, start, end, participant = 5
    print(f"    Features: {features_df.shape[1] - 5}")  
    print(f"    Saved: output/{subject_id}/features_raw.csv")
    
    print(f"\n    Label distribution:")
    for label, count in features_df['label'].value_counts().items():
        print(f"      {label:12s}: {count:4d} windows")
    
    
    # 5. Normalization
    print("[STEP 4/4] FEATURE NORMALIZATION")
    normalized_versions = normalize_all_versions(
        features_df,
        drop_cols=['label', 'participant', 'time_center', 'start', 'end']
    )
    
    for norm_type, norm_df in normalized_versions.items():
        out_path = f"output/{subject_id}/features_{norm_type}.csv"
        norm_df.to_csv(out_path, index=False)
        print(f" {norm_type:8s}: {out_path}")
    
    return features_df, normalized_versions


def process_all_wesad(wesad_path):
    
    valid_subjects = [f"S{i}" for i in range(2, 18) if i not in [1, 12]]
    
    print(f"PROCESSING ALL SUBJECTS")
    print(f"Valid subjects: {valid_subjects}")
    print(f"Total: {len(valid_subjects)} subjects")

    
    all_features_raw = []
    all_features_normalized = {'minmax': [], 'zscore': [], 'robust': []}
    
    for idx, subject_id in enumerate(valid_subjects, 1):
        print(f"\nSubject {idx}/{len(valid_subjects)}")
        
        subject_path = os.path.join(wesad_path, subject_id)
        
        if not os.path.exists(subject_path):
            print(f"[WARN] Subject not found: {subject_path}")
            continue
        
        try:
            features, normalized = process_wesad_subject(subject_id, wesad_path)
            
            all_features_raw.append(features)
            for norm_type in all_features_normalized.keys():
                if norm_type in normalized:
                    all_features_normalized[norm_type].append(normalized[norm_type])
            
            print(f"{subject_id} added to global dataset")
        
        except Exception as e:
            print(f"Failed to process {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("Global dataset created")
    
    if all_features_raw:
        df_all_raw = pd.concat(all_features_raw, ignore_index=True)
        df_all_raw.to_csv("output/wesad_all_features_raw.csv", index=False)
        
        print(f"Global raw features saved")
       
        print(f"    Shape: {df_all_raw.shape}")
        print(f"    Subjects: {df_all_raw['participant'].nunique()}")
        print(f"    Total windows: {len(df_all_raw)}")
        
        print(f"Global label distribution:")
        for label, count in df_all_raw['label'].value_counts().items():
            print(f"      {label:12s}: {count:5d} windows ({100*count/len(df_all_raw):.1f}%)")
        
        print(f"Normalized versions:")
        for norm_type, dfs in all_features_normalized.items():
            if dfs:
                df_all_norm = pd.concat(dfs, ignore_index=True)
                out_path = f"output/wesad_all_features_{norm_type}.csv"
                df_all_norm.to_csv(out_path, index=False)
                print(f"    {norm_type:8s}: {out_path} (Shape: {df_all_norm.shape})")
    else:
        print("No subjects processed successfully!")
        df_all_raw = None
        all_features_normalized = None
    
    return df_all_raw, all_features_normalized



if __name__ == "__main__":
    
    WESAD_PATH = "WESAD" 
    
    os.makedirs("output", exist_ok=True)
    process_all_wesad(WESAD_PATH)
    
    # single subject
    #print("\n[MODE] Single subject testing")
    
    #test_subject = "S2"
    #features, normalized = process_wesad_subject(test_subject, WESAD_PATH)
    
    
    
    
