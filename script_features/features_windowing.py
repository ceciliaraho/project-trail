# features_windowing.py
import numpy as np
import pandas as pd

from .features_breath import (
    resp_amplitude_features,
    resp_spectral_features,
    get_peaks_from_bf,
)
from .features_hr import hr_features_from_wave, get_peaks_from_hr
from .features_coupling import hr_bf_coupling

from .features_eda import eda_basic_features
from .features_emg import emg_basic_features


def extract_all_features(df, fs=700, window_s=10.0, hop_s=2.0):
    """
    features extraction from DataFrame from BF, HR, EDA, EMG with overlapping

    Args:
        df: DataFrame with ['label', 'time_from_start', ...]
             and:
              - 'BF' (es. %)
              - 'HR' in mV
              - 'EDA_uS' in µS
              - 'EMG_mV' in mV
        fs:  sampling frequency (Hz)
        window_s: time of windows in sec (default 10s)
        hop_s:    hop between windows in sec (default 2s → 80% overlap)
    """
    df = df.copy()
    window = int(fs * window_s)
    hop = int(fs * hop_s)
    out = []
    n = len(df)

    print(
        f"  Extracting features with {window_s:.1f}s windows, {hop_s:.1f}s hop "
        f"({window} samples window, {hop} samples hop @ {fs} Hz)"
    )

    available_signals = [c for c in df.columns if c in ["BF", "HR", "EDA", "EMG"]]
    print(f"  Available signals: {available_signals}")

    # Loop with overlap
    for start in range(0, n - window + 1, hop):
        end = start + window
        chunk = df.iloc[start:end]

        label_series = chunk["label"]
        if label_series.mode().empty:
            continue
        label = label_series.mode()[0]

        if str(label).lower() == "unlabeled":
            continue

        feats = {}

        # ================= RESPIRATION FEATURES (BF) =================
        if "BF" in chunk.columns:
            bf = chunk["BF"].values  
            
            feats.update(resp_amplitude_features(bf))

            bf_peaks, _ = get_peaks_from_bf(bf, label, fs)
            duration_s = window / fs

            if len(bf_peaks) > 1 and duration_s > 0:
                breaths = len(bf_peaks)
                resp_bpm = breaths * 60.0 / duration_s
                resp_hz = resp_bpm / 60.0
            else:
                breaths = len(bf_peaks)
                resp_bpm = np.nan
                resp_hz = np.nan

            feats["resp_peak_count"] = int(breaths)
            feats["resp_bpm_from_peaks"] = (
                float(resp_bpm) if not np.isnan(resp_bpm) else np.nan
            )
            feats["resp_peak_freq_hz"] = (
                float(resp_hz) if not np.isnan(resp_hz) else np.nan
            )

            feats.update(resp_spectral_features(bf, fs))

        # ================= HR / ECG FEATURES =================
        if "HR" in chunk.columns:
            hr = chunk["HR"].values  
            feats.update(hr_features_from_wave(hr, fs=fs))

        # ================= COUPLING BF–HR =================
        if "BF" in chunk.columns and "HR" in chunk.columns:
            feats.update(hr_bf_coupling(hr, bf, fs, max_lag_s=2.0))

        # ================= EDA FEATURES =================
        if "EDA" in chunk.columns:
            try:
                eda_feats = eda_basic_features(chunk["EDA"].values, fs=fs)
                feats.update(eda_feats)
            except Exception as e:
                print(
                    f"[WARN] EDA feature extraction failed at window {start}:{end}: {e}"
                )

        # ================= EMG FEATURES =================
        if "EMG" in chunk.columns:
            try:
                emg_feats = emg_basic_features(chunk["EMG"].values, fs=fs)
                feats.update(emg_feats)
            except Exception as e:
                print(
                    f"[WARN] EMG feature extraction failed at window {start}:{end}: {e}"
                )

        # ================= METADATA =================
        feats["label"] = label
        feats["start"] = float(start / fs)
        feats["time_center"] = float((start / fs) + window_s / 2.0)
        feats["end"] = float(end / fs)

        if "participant" in df.columns:
            part_series = chunk["participant"]
            if not part_series.mode().empty:
                feats["participant"] = part_series.mode()[0]

        out.append(feats)

    df_features = pd.DataFrame(out)

    if df_features.empty:
        print(" No feature vectors extracted (check labels and input df).")
        return df_features

    # Replace inf/-inf with NaN, after fillna with 0
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    meta_cols = ["label", "time_center", "start", "end"]
    if "participant" in df_features.columns:
        meta_cols.append("participant")
    feat_cols = [c for c in df_features.columns if c not in meta_cols]
    df_features = df_features[feat_cols + meta_cols]

    print(
        f"  Extracted {len(df_features)} feature vectors ({len(feat_cols)} features each)"
    )

    return df_features
