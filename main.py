# Refactored robust waveform analysis script

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

FOLDER_PATH = r'/Users/jonghyeokkim/PycharmProjects/JitterAnalysisTool_Rohde_ver/Data/JitterTest_260331-0401'
OUTPUT_DIR = r'.\\output\\260331-0401_Delaytime_jitter_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNEL_LIST = ['CH1', 'CH2']
records = []
DEBUG = True

# ==============================
# Smart Header Detection
# ==============================
def find_header_line(path, max_lines=100):
    with open(path, 'r', encoding='latin1', errors='ignore') as f:
        for i, line in enumerate(f):
            line_upper = line.upper()
            if ('TIME' in line_upper) and ('CH' in line_upper):
                sep = ';' if line.count(';') >= line.count(',') else ','
                return i, sep
            if i > max_lines:
                break
    raise ValueError(f"Header not found in {path}")

# ==============================
# Column Normalization
# ==============================
def normalize_columns(df):
    cols = []
    for c in df.columns:
        c = str(c).replace("\ufeff", "").strip()
        c = re.sub(r"\s+", "", c)
        c = re.sub(r"\(.*?\)", "", c)
        c = c.upper().replace("CHANNEL", "CH")
        cols.append(c)

    df.columns = cols

    # fallback: UNNAMED → TIME
    if 'TIME' not in df.columns:
        unnamed = [c for c in df.columns if 'UNNAMED' in c]
        if unnamed:
            df.rename(columns={unnamed[0]: 'TIME'}, inplace=True)

    return df

# ==============================
# Robust CSV Reader
# ==============================
def read_waveform_csv(path):
    header_idx, sep = find_header_line(path)

    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'latin1']:
        try:
            df = pd.read_csv(path, skiprows=header_idx, sep=sep, encoding=enc, engine='python')
            df = normalize_columns(df)

            if 'TIME' not in df.columns:
                continue

            df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')

            for ch in CHANNEL_LIST:
                if ch in df.columns:
                    df[ch] = pd.to_numeric(df[ch], errors='coerce')

            df = df.dropna(subset=['TIME']).reset_index(drop=True)

            if len(df) == 0:
                continue

            return df

        except Exception:
            continue

    raise ValueError(f"CSV parsing failed: {path}")

# ==============================
# Waveform Analysis
# ==============================
def analyze_waveform(file_path, filename):
    if DEBUG:
        print(f"Processing: {filename}")

    df = read_waveform_csv(file_path)
    threshold = 3

    rising_indices = {}
    for ch in CHANNEL_LIST:
        if ch in df.columns:
            idxs = df.index[df[ch] >= threshold]
            rising_indices[ch] = idxs[0] if len(idxs) > 0 else None
        else:
            rising_indices[ch] = None

    if rising_indices['CH1'] is None:
        if DEBUG:
            print("No CH1 trigger found")
        return

    t_ref = df.loc[rising_indices['CH1'], 'TIME']

    for ch in CHANNEL_LIST[1:]:
        if rising_indices[ch] is not None:
            t_val = df.loc[rising_indices[ch], 'TIME']
            delay = (t_val - t_ref) * 1e6

            records.append({
                'Filename': filename,
                'Channel': ch,
                'Delay (us)': delay
            })

    # Plot waveform
    plt.figure()
    for ch in CHANNEL_LIST:
        if ch in df.columns:
            plt.plot(df['TIME'], df[ch], label=ch)

    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, filename.replace('.csv', '_waveform.png')))
    plt.close()

# ==============================
# Main Execution
# ==============================
files = [f for f in os.listdir(FOLDER_PATH) if f.startswith('Waveform') and f.endswith('.csv')]

for f in files:
    analyze_waveform(os.path.join(FOLDER_PATH, f), f)

# ==============================
# Result Processing
# ==============================
df_result = pd.DataFrame(records)

if not df_result.empty:
    df_result.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)

    df_stats = df_result.groupby('Channel')['Delay (us)'].agg(['mean', 'std', 'min', 'max'])
    df_stats['count'] = df_result.groupby('Channel').size()
    df_stats['range'] = df_stats['max'] - df_stats['min']
    df_stats['cv'] = df_stats['std'] / df_stats['mean']

    df_stats.to_csv(os.path.join(OUTPUT_DIR, 'stats.csv'))

print("Done.")
