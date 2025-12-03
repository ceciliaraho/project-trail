# script_features/features_emg.py

import numpy as np

def emg_basic_features(x, fs=700):
    x = np.asarray(x, dtype=float)

    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {
            "emg_rms": np.nan,
            "emg_mav": np.nan,
            "emg_var": np.nan,
            "emg_iemg": np.nan,
            "emg_wl": np.nan,
            "emg_zc_rate": np.nan,
        }

    # RMS
    rms = float(np.sqrt(np.mean(x ** 2)))

    # Mean Absolute Value
    mav = float(np.mean(np.abs(x)))

    # variance
    var = float(np.var(x))

    # Integrated EMG 
    iemg = float(np.sum(np.abs(x)))

    # Waveform length
    wl = float(np.sum(np.abs(np.diff(x))))

    # Zero Crossing rate 
    thr = 0.01  # mV (I can change)
    x1 = x[:-1]
    x2 = x[1:]
    crossings = ((x1 * x2) < 0) & (np.abs(x1 - x2) > thr)
    zc_count = np.sum(crossings)
    zc_rate = float(zc_count / (len(x) / fs)) 

    return {
        "emg_rms": rms,
        "emg_mav": mav,
        "emg_var": var,
        "emg_iemg": iemg,
        "emg_wl": wl,
        "emg_zc_rate": zc_rate,
    }
