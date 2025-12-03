# features_debug.py
import numpy as np
import matplotlib.pyplot as plt
from .features_breath import get_peaks_from_bf
from .features_hr import get_peaks_from_hr

def plot_peaks_by_section(df, fs=700):
    
    
    unique_labels = df["label"].dropna().unique()
    
    for label in unique_labels:
        if str(label).lower() == "unlabeled":
            continue
        
        section_df = df[df["label"] == label].copy()
        if section_df.empty:
            continue
        
        bf = section_df["BF"].values
        hr = section_df["HR"].values
        time = section_df.get("time_from_start", np.arange(len(bf)) / fs).values
        
    
        bf_peaks, bf_smooth = get_peaks_from_bf(bf, label, fs)
        hr_peaks, hr_smooth = get_peaks_from_hr(hr, fs)
        
      
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
        
        # BF plot
        ax1.plot(time, bf_smooth, label="Respiration smoothed")
        if len(bf_peaks) > 0:
            ax1.plot(time[bf_peaks], bf_smooth[bf_peaks], "x", 
                     label=f"BF Peaks ({len(bf_peaks)})")
        ax1.set_title(f"{label} - Breathing Signal")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("BF (physical units, es. %)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # HR plot
        ax2.plot(time, hr_smooth, label="ECG smoothed")
        if len(hr_peaks) > 0:
            ax2.plot(time[hr_peaks], hr_smooth[hr_peaks], "o",
                     label=f"HR Peaks ({len(hr_peaks)})")
        ax2.set_title(f"{label} - Heart Signal")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("HR (mV)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

