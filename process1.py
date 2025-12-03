import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler


def uniform_amplitude(signal, target_std=0.1):
    signal = signal - np.mean(signal)
    current_std = np.std(signal)
    if current_std < 1e-6:
        return signal
    return signal * (target_std / current_std)


def estimate_cutoff_from_variance(
    signal,
    low_std=0.5,
    high_std=5.0,
    low_cutoff=0.2,
    high_cutoff=0.7,
):
    """
    cutoff frequency on signal variance 
    
    """
    std = np.std(signal)
    if std <= low_std:
        return low_cutoff
    elif std >= high_std:
        return high_cutoff
    else:
        ratio = (std - low_std) / (high_std - low_std)
        return low_cutoff + ratio * (high_cutoff - low_cutoff)


def lowpass_filter(data, fs, cutoff=0.5, order=2):
    
    data = np.nan_to_num(data)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99

    b, a = butter(order, normal_cutoff, btype="low", analog=False)


    if len(data) < 3 * max(len(a), len(b)):
        return data

    return filtfilt(b, a, data)


def highpass_filter(data, fs, cutoff=0.5, order=2):
    
    # useful for remove component DC (es. EMG).

    data = np.nan_to_num(data)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99

    b, a = butter(order, normal_cutoff, btype="high", analog=False)

    if len(data) < 3 * max(len(a), len(b)):
        return data

    return filtfilt(b, a, data)


def bandpass_filter(data, fs, lowcut=20.0, highcut=250.0, order=2):
    # useful for EMG 
    
    data = np.nan_to_num(data)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1.0:
        high = 0.99
    if low <= 0.0:
        low = 0.001

    if low >= high:

        return highpass_filter(data, fs, cutoff=lowcut, order=order)

    b, a = butter(order, [low, high], btype="band", analog=False)

    if len(data) < 3 * max(len(a), len(b)):
        return data

    return filtfilt(b, a, data)


def zscore_normalize(signal):

    scaler = StandardScaler()
    if isinstance(signal, pd.Series):
        signal = signal.values
    return scaler.fit_transform(signal.reshape(-1, 1)).flatten()


def min_max_normalize(signal):
    if isinstance(signal, pd.Series):
        signal = signal.values
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return signal
    return (signal - min_val) / (max_val - min_val)


# =========================================================
# PREPROCESSING BF (RESPIRATION)
# =========================================================

def preprocess_breath_signals(df, fs_custom=700):

    df = df.copy()

    if "BF" not in df.columns:
        print("Column 'BF' not found in DataFrame")
        return df


    # missed value interpolation
    df["BF"] = df["BF"].interpolate(method="linear", limit_direction="both")

    cutoff_dynamic = estimate_cutoff_from_variance(
        df["BF"].values,
        low_std=0.5,  # 0.5% std bassa
        high_std=5.0,  # 5% std alta
    )
    print(
        f"[BF] Cutoff estimated: {cutoff_dynamic:.2f} Hz "
        f"(based on std={df['BF'].std():.3f}%)"
    )

    df["BF"] = lowpass_filter(df["BF"].values, fs=fs_custom, cutoff=cutoff_dynamic)

    return df


# =========================================================
# PREPROCESSING HR (ECG)
# =========================================================

def preprocess_hr_signals(df, columns=["HR"]):

    df = df.copy()

    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' not found")
            continue

        signal = df[column].copy()

        signal = signal.interpolate(method="linear", limit_direction="both")

        # Smoothing
        window_size = max(3, int(700 * 0.005))  # ~5ms window
        signal = signal.rolling(window=window_size, center=True, min_periods=1).mean()

        print(f"[{column}] After smoothing: [{signal.min():.3f}, {signal.max():.3f}] mV")
        print(f"[{column}] ✓ Kept in physical units (no normalization)")

        df[column] = signal

    return df


# =========================================================
# PREPROCESSING EDA
# =========================================================

def preprocess_eda_signals(df, columns=["EDA"], fs_custom=700):
    
    df = df.copy()

    for column in columns:
        if column not in df.columns:
            print(f"[WARN] Column '{column}' not found for EDA")
            continue

        print(f"[{column}] Input range: [{df[column].min():.3f}, {df[column].max():.3f}] µS")

        sig = df[column].astype(float)

        sig = sig.interpolate(method="linear", limit_direction="both")

        sig = lowpass_filter(sig.values, fs=fs_custom, cutoff=2.0)

        # no negative valued (EDA cannot be < 0)
        sig = np.clip(sig, 0.0, None)

        df[column] = sig

    return df


# =========================================================
# PREPROCESSING EMG
# =========================================================

def preprocess_emg_signals(df, columns=["EMG"], fs_custom=700):
    df = df.copy()

    for column in columns:
        if column not in df.columns:
            print(f"[WARN] Column '{column}' not found for EMG")
            continue

        print(f"[{column}] Input range: [{df[column].min():.3f}, {df[column].max():.3f}] mV")

        sig = df[column].astype(float)

        sig = sig.interpolate(method="linear", limit_direction="both")

        sig_filt = bandpass_filter(sig.values, fs=fs_custom, lowcut=20.0, highcut=250.0, order=2)

        df[column] = sig_filt

    return df


if __name__ == "__main__":
    #test, Create dummy data
    n = 1000
    test_df = pd.DataFrame({
        "BF":  np.random.randn(n) * 5 + 10,
        "HR":  np.random.randn(n) * 0.3 + 0.5,
        "EDA": np.abs(np.random.randn(n) * 1.0 + 3.0),
        "EMG": np.random.randn(n) * 0.05,
        "label": ["baseline"] * n,
    })

    # Preprocess
    test_df = preprocess_breath_signals(test_df, fs_custom=700)
    test_df = preprocess_hr_signals(test_df)
    test_df = preprocess_eda_signals(test_df)
    test_df = preprocess_emg_signals(test_df)

    print("Preprocessing test passed")
