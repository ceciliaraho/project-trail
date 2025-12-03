import pandas as pd
import numpy as np
import os

# Constant values
CHAN_BIT = 2**16  # 65536 (16-bit ADC)
VCC = 3.0


def convert_ecg_raw_to_mv(signal_raw):
    """ECG raw (0-65535) in millivolt"""
    return ((signal_raw / CHAN_BIT - 0.5) * VCC)

def convert_eda_raw_to_µS(signal_raw):
    """EDA raw in microSiemens"""
    return (((signal_raw / CHAN_BIT) * VCC)/0.12)

def convert_resp_raw_to_percent(signal_raw):
    """ RESP raw (0-65535) in %"""
    return (signal_raw / CHAN_BIT - 0.5) * 100

def convert_emg_raw_to_mv(signal_raw):
    """EMG raw (0-65535) in millivolt"""
    return (signal_raw / CHAN_BIT - 0.5) * VCC


def load_wesad_respiban_txt(subject_path):
    """
    Load data

    """
    subject_id = os.path.basename(subject_path)
    txt_file = os.path.join(subject_path, f"{subject_id}_respiban.txt")
    
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File not found: {txt_file}")
    
    # header JSON 
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    # Find data
    data_start_line = 0
    for i, line in enumerate(lines):
        if '# EndOfHeader' in line or 'EndOfHeader' in line:
            data_start_line = i + 1
            break
        if line.strip() and not line.startswith('#'):
            try:
                parts = line.strip().split()
                if len(parts) >= 10 and parts[0].isdigit():
                    data_start_line = i
                    break
            except:
                continue
    
    # Read data
    df_raw = pd.read_csv(txt_file, sep='\s+', header=None, skiprows=data_start_line, 
                         engine='python', on_bad_lines='skip')
    
    
    # 10 columns: line_num, ignore, 8 sensors
    if df_raw.shape[1] >= 10:
        df_raw = df_raw.iloc[:, :10]
        df_raw.columns = ['line_num', 'ignore', 'ECG_raw', 'EDA_raw', 'EMG_raw', 
                          'TEMP_raw', 'ACC_X_raw', 'ACC_Y_raw', 'ACC_Z_raw', 'RESP_raw']
    else:
        raise ValueError(f"Expected at least 10 columns, got {df_raw.shape[1]}")
    
    # Convert raw values
    print("Converting raw values to physical units")
    df_converted = pd.DataFrame()
    fs_original = 700  # Hz
    df_converted['time_s'] = np.arange(len(df_raw)) / fs_original
    
    df_converted['ECG_mv'] = convert_ecg_raw_to_mv(df_raw['ECG_raw'].values)
    df_converted['RESP_pct'] = convert_resp_raw_to_percent(df_raw['RESP_raw'].values)
    df_converted['EDA_uS']   = convert_eda_raw_to_µS(df_raw['EDA_raw'].values)
    df_converted['EMG_mV']   = convert_emg_raw_to_mv(df_raw['EMG_raw'].values)
    
    return df_converted


def load_wesad_labels(subject_path):
    """
    labels from SX_quest.csv.

    """
    subject_id = os.path.basename(subject_path)
    quest_file = os.path.join(subject_path, f"{subject_id}_quest.csv")
    
    if not os.path.exists(quest_file):
        raise FileNotFoundError(f"File not found: {quest_file}")

    
    # read file
    with open(quest_file, 'r') as f:
        lines = f.readlines()
    
    order_line = None
    start_line = None
    end_line = None
    
    # find lines with ORDER, START, END
    for line in lines:
        line_clean = line.strip()
        if line_clean.startswith('# ORDER'):
            order_line = line_clean
        elif line_clean.startswith('# START'):
            start_line = line_clean
        elif line_clean.startswith('# END'):
            end_line = line_clean
    
    if not order_line or not start_line or not end_line:
        raise ValueError("Could not find ORDER, START, END lines in the file")
    
    # Parse
    def parse_line(line):
        # remove prefix (# ORDER, # START, # END)
        line = line.split(';', 1)[1] if ';' in line else line
        # Split with ;
        values = [x.strip() for x in line.split(';') if x.strip()]
        return values
    
    conditions = parse_line(order_line)
    start_times = parse_line(start_line)
    end_times = parse_line(end_line)
    
    # Convert time from minutes to sec
    def parse_time(time_str):
        try:
            if not time_str or time_str == '':
                return np.nan
            minutes = float(time_str)
            return minutes * 60.0
        except:
            return np.nan
    
    # Mappa nomi condizioni
    name_map = {
        'base': 'baseline',
        'baseline': 'baseline',
        'tsst': 'stress',
        'stress': 'stress',
        'fun': 'amusement',
        'amusement': 'amusement',
        'medi 1': 'meditation',
        'medi 2': 'meditation',
        'medi1': 'meditation',
        'medi2': 'meditation',
        'meditation': 'meditation',
        'meditation 1': 'meditation',
        'meditation 2': 'meditation'
    }
    
    # label list
    labels_info = []
    for condition, start_str, end_str in zip(conditions, start_times, end_times):
        condition_lower = str(condition).strip().lower()
        
        if 'read' in condition_lower:
            continue
        
        label_name = name_map.get(condition_lower, condition_lower)
        
        start_s = parse_time(start_str)
        end_s = parse_time(end_str)
        
        if not np.isnan(start_s) and not np.isnan(end_s):
            labels_info.append({
                'condition': label_name,
                'start_s': start_s,
                'end_s': end_s
            })
    
    df_labels = pd.DataFrame(labels_info)
    
    if len(df_labels) == 0:
        raise ValueError("No valid labels found")
    
    print(f"Found {len(df_labels)} labeled conditions:")
    for _, row in df_labels.iterrows():
        duration = row['end_s'] - row['start_s']
        print(f"{row['condition']:15s}: {row['start_s']:7.1f}s -> {row['end_s']:7.1f}s  (duration: {duration:.1f}s)")
    
    return df_labels


def merge_signals_and_labels(df_signals, df_labels):
    """Merge segnali con labels."""
    
    df_signals = df_signals.copy()
    df_signals['label'] = 'unlabeled'
    
    for _, label_row in df_labels.iterrows():
        mask = (df_signals['time_s'] >= label_row['start_s']) & \
               (df_signals['time_s'] < label_row['end_s'])
        df_signals.loc[mask, 'label'] = label_row['condition']
    
    # remove unlabeled
    df_signals = df_signals[df_signals['label'] != 'unlabeled'].copy()
    df_signals.reset_index(drop=True, inplace=True)
    
    print(f"\nLabel distribution after merging:")
    print(df_signals['label'].value_counts())
    
    return df_signals


def load_wesad_subject(subject_path):
    
    subject_id = os.path.basename(subject_path)
    print(f"\n{'='*60}")
    print(f"Loading WESAD Subject: {subject_id}")
    print(f"{'='*60}\n")
    
    # 1. load signals
    df_signals = load_wesad_respiban_txt(subject_path)
    
    # 2. load labels
    df_labels = load_wesad_labels(subject_path)
    
    # 3. Merge
    df_merged = merge_signals_and_labels(df_signals, df_labels)
    

    df_final = pd.DataFrame()
    df_final['time_from_start'] = df_merged['time_s'].values
    df_final['BF'] = df_merged['RESP_pct'].values  
    df_final['HR'] = df_merged['ECG_mv'].values    
    df_final['EDA']  = df_merged['EDA_uS'].values
    df_final['EMG']  = df_merged['EMG_mV'].values
    df_final['label'] = df_merged['label'].values
    
    return df_final


if __name__ == "__main__":
    # Test 
    WESAD_PATH = "WESAD" 
    
    subject_id = "S2"
    subject_path = os.path.join(WESAD_PATH, subject_id)

    df = load_wesad_subject(subject_path)
    
    print("\n" + "="*60)
    print("DATAFRAME PREVIEW")
    print("="*60)
    print(df.head(10))
    print("\nColumns:", df.columns.tolist())
    print("\nShape:", df.shape)
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print("\nSaving test output...")
    df.to_csv("test_wesad_S2.csv", index=False)
    print("Saved: test_wesad_S2.csv")