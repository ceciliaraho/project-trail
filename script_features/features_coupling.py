# features_coupling.py
import numpy as np
from scipy.signal import correlate

def hr_bf_coupling(hr, bf, fs=700, max_lag_s=2.0):
    """
    coupling between HR (ECG waveform) and BF (resp waveform)
    with normalized cross-correlation
    """
    hr0 = np.asarray(hr) - np.mean(hr)
    bf0 = np.asarray(bf) - np.mean(bf)
    
    max_lag = int(max_lag_s * fs)
    
    # Cross-correlation
    xcorr = correlate(hr0, bf0, mode="full")
    lags = np.arange(-len(hr0)+1, len(bf0))
    
    m = (lags >= -max_lag) & (lags <= max_lag)
    xcorr = xcorr[m]
    lags = lags[m]
    
    # Normalization
    norm = (np.std(hr0) * np.std(bf0) * len(hr0) + 1e-12)
    xcorr = xcorr / norm
    
    idx = int(np.argmax(np.abs(xcorr)))
    
    return {
        "hr_bf_xcorr_max": float(xcorr[idx]),
        "hr_bf_xcorr_lag_s": float(lags[idx] / fs),
    }


