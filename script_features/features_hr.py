# features_hr.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, correlate, welch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def _filter_intervals_by_rate(peak_idx, fs, min_bpm, max_bpm):
    """
    filter peaks to delete intervals RR to much steep or slow .
    """
    peak_idx = np.asarray(peak_idx, dtype=int)
    if len(peak_idx) < 2:
        return peak_idx
    
    ibi = np.diff(peak_idx) / fs  # inter-beat interval in sec
    bpm = 60.0 / np.clip(ibi, 1e-6, None)
    keep = (bpm >= min_bpm) & (bpm <= max_bpm)
    
    return np.array(
        [peak_idx[0]] + [peak_idx[i+1] for i, k in enumerate(keep) if k],
        dtype=int
    )

def get_peaks_from_hr(hr, fs=700):
    """
    Peaks detection
    """
    hr = np.asarray(hr)
    hr_smooth = gaussian_filter1d(hr, sigma=1.0)
    
    hr_range = float(np.max(hr_smooth) - np.min(hr_smooth))
    hr_std = float(np.std(hr_smooth))
    
    if hr_range < 0.5 or hr_std < 0.2:
        distance = int(fs * 0.30)
        prom = 0.05
    elif hr_range < 1.5 or hr_std < 0.5:
        distance = int(fs * 0.50)
        prom = 0.10
    else:
        distance = int(fs * 0.70)
        prom = 0.15
    
    peaks, _ = find_peaks(hr_smooth, distance=distance, prominence=prom)
    
    peaks = _filter_intervals_by_rate(peaks, fs, min_bpm=30, max_bpm=180)
    
    return peaks, hr_smooth


def hr_features_from_wave(hr, fs=700):

    hr = np.asarray(hr)
    
    feats = {
        "ecg_mean": float(np.mean(hr)),
        "ecg_std": float(np.std(hr)),
        "ecg_min": float(np.min(hr)),
        "ecg_max": float(np.max(hr)),
        "ecg_range": float(np.max(hr) - np.min(hr)),
        "ecg_skew": float(skew(hr)) if len(hr) > 2 else np.nan,
        "ecg_kurtosis": float(kurtosis(hr)) if len(hr) > 2 else np.nan,
        "ecg_slope": float((hr[-1] - hr[0]) / len(hr)) if len(hr) > 1 else np.nan,
    }
    
    peaks, _ = get_peaks_from_hr(hr, fs)
    
    feats["ecg_low_peaks_flag"] = int(len(peaks) < 2)
    
    if len(peaks) > 1:
        # Inter-beat intervals (RR intervals)
        rr = np.diff(peaks) / fs  # in secondi
        rr_ms = rr * 1000.0       # in millisecondi
        
        mean_rr = float(np.mean(rr)) if len(rr) > 0 else np.nan
        
        # HRV time-domain metrics
        feats.update({
            "hr_peak_count": int(len(peaks)),
            "hr_ibi_mean_s": float(np.mean(rr)),
            "hr_ibi_std_s": float(np.std(rr)),
            "hr_bpm_from_peaks": float(60.0 / mean_rr) if (mean_rr > 0 and not np.isnan(mean_rr)) else np.nan,
            "hr_sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan,
            "hr_rmssd_ms": float(np.sqrt(np.mean(np.diff(rr_ms)**2))) if len(rr_ms) > 1 else np.nan,
            "hr_pnn50": float(np.mean(np.abs(np.diff(rr_ms)) > 50.0)) if len(rr_ms) > 1 else np.nan,
        })
    else:
        feats.update({
            "hr_peak_count": 0,
            "hr_ibi_mean_s": np.nan,
            "hr_ibi_std_s": np.nan,
            "hr_bpm_from_peaks": np.nan,
            "hr_sdnn_ms": np.nan,
            "hr_rmssd_ms": np.nan,
            "hr_pnn50": np.nan
        })
    
    return feats
