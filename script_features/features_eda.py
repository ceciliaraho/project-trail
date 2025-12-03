# script_features/features_eda.py

import numpy as np
from scipy.signal import find_peaks

def eda_basic_features(x, fs=700):
    x = np.asarray(x, dtype=float)

    # sicurezza su NaN/inf
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {
            "eda_mean": np.nan,
            "eda_median": np.nan,
            "eda_std": np.nan,
            "eda_var": np.nan,
            "eda_min": np.nan,
            "eda_max": np.nan,
            "eda_iqr": np.nan,
            "eda_range": np.nan,
            "eda_slope": np.nan,
            "eda_auc": np.nan,
            "eda_peak_count": np.nan,
            "eda_peak_mean_amp": np.nan,
        }

    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x))
    var = float(np.var(x))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    value_range = float(x_max - x_min)

    # trend (slope)
    t = np.linspace(0, 1, len(x))
    coeffs = np.polyfit(t, x, 1)
    slope = float(coeffs[0])

    # area behind slope 
    auc = float(np.trapz(x, dx=1.0 / fs))

    # “attivation”
    dx = np.diff(x) * fs  
    # easy threshold semplice: 0.05 µS/s (I can change)
    thr = max(0.05, 0.1 * np.std(dx))
    peaks, props = find_peaks(dx, height=thr, distance=int(0.5 * fs)) 

    peak_count = len(peaks)
    if peak_count > 0:
        peak_mean_amp = float(np.mean(props["peak_heights"]))
    else:
        peak_mean_amp = 0.0

    return {
        "eda_mean": mean,
        "eda_median": median,
        "eda_std": std,
        "eda_var": var,
        "eda_min": x_min,
        "eda_max": x_max,
        "eda_iqr": iqr,
        "eda_range": value_range,
        "eda_slope": slope,
        "eda_auc": auc,
        "eda_peak_count": float(peak_count),
        "eda_peak_mean_amp": peak_mean_amp,
    }
