# features_breath.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, correlate, welch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def get_peaks_from_bf(bf, label=None, fs=700):
    """
    Find peaks
    BF is the waveform
    """
    bf = np.asarray(bf)
    bf_smooth = gaussian_filter1d(bf, sigma=0.8)
    
    bf_range = np.max(bf_smooth) - np.min(bf_smooth)
    bf_std = np.std(bf_smooth)
    
    # Adaptive parameters
    if bf_range < 0.1 or bf_std < 0.02:
        distance = int(fs * 0.1)
        prom = 0.01
    elif bf_range < 0.2 or bf_std < 0.04:
        distance = int(fs * 0.2)
        prom = 0.03
    else:
        distance = int(fs * 0.35)
        prom = 0.05
    
    peaks, _ = find_peaks(bf_smooth, distance=distance, prominence=prom)
    return peaks, bf_smooth

def resp_amplitude_features(bf, fs=700):
    """
    resp_amp_* = amplitude
    """
    bf = np.asarray(bf)

    return {
        "resp_amp_mean": float(np.mean(bf)),
        "resp_amp_std": float(np.std(bf)),
        "resp_amp_min": float(np.min(bf)),
        "resp_amp_max": float(np.max(bf)),
        "resp_amp_range": float(np.max(bf) - np.min(bf)),
        "resp_amp_skew": float(skew(bf)) if len(bf) > 2 else np.nan,
        "resp_amp_kurtosis": float(kurtosis(bf)) if len(bf) > 2 else np.nan,
    }

def resp_spectral_features(bf, fs=700):
    bf = np.asarray(bf)

    if len(bf) < 4:
        return {
            "resp_spec_centroid_hz": 0.0,
            "resp_spec_entropy": 0.0,
        }

    # Welch periodogram
    f, Pxx = welch(bf, fs=fs, nperseg=min(len(bf), 1024))
    if len(f) == 0 or np.all(Pxx == 0):
        return {
            "resp_spec_centroid_hz": 0.0,
            "resp_spec_entropy": 0.0,
        }

    total = float(np.trapz(Pxx, f) + 1e-12)

    # Spectral centroid
    centroid = float(np.sum(f * Pxx) / total)

    # Spectral entropy
    Pn = Pxx / (np.sum(Pxx) + 1e-12)
    spec_ent = float(entropy(Pn) / np.log(len(Pn)))

    return {
        "resp_spec_centroid_hz": centroid,
        "resp_spec_entropy": spec_ent,
    }