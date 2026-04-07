# Refactored robust waveform analysis script

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import shutil

FOLDER_PATH = r'D:\work\02_EIS\01_Seismic\04_지연편차 측정\260331-0401_Delaytime_jitter'
OUTPUT_DIR = os.path.join('output', '260331-0401_Delaytime_jitter_output')
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
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
            # More flexible detection: contains 'CH1', 'CH2', or both 'TIME' and 'CH'
            has_channels = any(ch in line_upper for ch in CHANNEL_LIST)
            has_time_ch = ('TIME' in line_upper) and ('CH' in line_upper)
            
            if has_channels or has_time_ch:
                sep = ';' if line.count(';') >= line.count(',') else ','
                return i, sep
            if i > max_lines:
                break
    raise ValueError(f"Header not found in {path}")

# ==============================
# Metadata Extraction
# ==============================
def extract_metadata(path):
    meta = {'interval': None, 'position': 0, 'reference': 50}
    with open(path, 'r', encoding='latin1', errors='ignore') as f:
        for i, line in enumerate(f):
            if i > 100: break
            line = line.strip()
            if not line: continue
            
            # Use comma or semicolon separator for metadata
            parts = re.split('[;,]', line)
            key = parts[0].strip().upper()
            
            if 'SAMPLE INTERVAL' in key and len(parts) > 1:
                try: meta['interval'] = float(parts[1])
                except: pass
            elif 'HORIZONTAL POSITION' in key and len(parts) > 1:
                try: meta['position'] = float(parts[1])
                except: pass
            elif 'REFERENCE POINT' in key and len(parts) > 1:
                # e.g., '50 %'
                val = parts[1].replace('%', '').strip()
                try: meta['reference'] = float(val)
                except: pass
    return meta

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
    meta = extract_metadata(path)

    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'latin1']:
        try:
            df = pd.read_csv(path, skiprows=header_idx, sep=sep, encoding=enc, engine='python')
            df = normalize_columns(df)

            # Robust time column handling
            if 'TIME' not in df.columns:
                # If TIME is missing, see if index 0 is Unnamed or empty
                unnamed = [c for c in df.columns if 'UNNAMED' in c]
                if unnamed:
                    df.rename(columns={unnamed[0]: 'TIME'}, inplace=True)
                else:
                    # Insert a placeholder TIME column
                    df.insert(0, 'TIME', float('nan'))

            # Convert to numeric
            df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
            for ch in CHANNEL_LIST:
                if ch in df.columns:
                    df[ch] = pd.to_numeric(df[ch], errors='coerce')

            # Reconstruction of TIME if it's all NaN and we have metadata
            if df['TIME'].isna().all() and meta['interval'] is not None:
                if DEBUG:
                    print(f"  Reconstructing TIME axis for {os.path.basename(path)}")
                
                # Formula: Center is at 'position'. Trigger is at 0.0 usually.
                # First point = position - (CenterIndex * interval)
                # But a simpler way if absolute time doesn't matter:
                # index * interval
                num_points = len(df)
                start_time = meta['position'] - (num_points / 2 * meta['interval'])
                df['TIME'] = start_time + (df.index * meta['interval'])

            df = df.dropna(subset=['TIME']).reset_index(drop=True)

            if len(df) == 0:
                continue

            return df

        except Exception as e:
            if DEBUG:
                print(f"  Error reading {path} with {enc}: {e}")
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

    if rising_indices['CH1'] is None or rising_indices['CH2'] is None:
        if DEBUG:
            print(f"Skipping {filename}: Trigger missing on CH1 or CH2")
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
            # Plot the line
            line, = plt.plot(df['TIME'], df[ch], label=ch, alpha=0.7)
            
            # Plot the detected trigger point with a marker
            idx = rising_indices[ch]
            if idx is not None:
                plt.plot(df.loc[idx, 'TIME'], df.loc[idx, ch], 'o', color=line.get_color(), markersize=8)
                plt.annotate(f"{ch} Trigger", (df.loc[idx, 'TIME'], df.loc[idx, ch]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.legend()
    plt.grid(True)
    plt.title(f"Waveform Analysis: {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
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

    # ==============================
    # Summary Plotting
    # ==============================
    # 1. Trend Plot: Delay Time vs Filename
    # Sort by filename to ensure chronological order based on Waveform_YYYY-MM-DD...
    df_plot = df_result.sort_values('Filename')
    
    plt.figure(figsize=(30, 16))
    for ch in df_plot['Channel'].unique():
        ch_data = df_plot[df_plot['Channel'] == ch]
        plt.plot(ch_data['Filename'], ch_data['Delay (us)'], marker='o', markersize=4, linestyle='-', label=ch, alpha=0.8)

        mean_val = ch_data['Delay (us)'].mean()
        plt.axhline(mean_val, color='red', linestyle='dashed', linewidth=1)
        plt.text(plt.xlim()[1] * 0.9, mean_val, f'Mean: {mean_val:.2f}')

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("Delay Time Trend (us) across Files", fontsize=14)
    plt.xlabel("Filename", fontsize=3)
    plt.ylabel("Delay (us)", fontsize=12)
    plt.legend(['Delay time(CH1->CH2)'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_delay_trend.png'), dpi=150)
    plt.close()

    if DEBUG:
        print(f"  Summary trend plot saved to {os.path.join(OUTPUT_DIR, 'summary_delay_trend.png')}")

    # 2. Distribution Plot: Histogram
    plt.figure(figsize=(10, 6))
    for ch in df_result['Channel'].unique():
        ch_data = df_result[df_result['Channel'] == ch]
        plt.hist(ch_data['Delay (us)'], bins='auto', alpha=0.7, label=ch, edgecolor='black', density=False)
        
        # Add labels for statistics
        mean_val = ch_data['Delay (us)'].mean()
        std_val = ch_data['Delay (us)'].std()
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
        plt.text(mean_val, plt.ylim()[1] * 0.9, f'Mean: {mean_val:.3f} us\nStd: {std_val:.3f} us', color='red')

    plt.title("Delay Time Distribution (Histogram)", fontsize=14)
    plt.xlabel("Delay (us)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(['Delay time(CH1->CH2)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_delay_distribution.png'), dpi=150)
    plt.close()

    if DEBUG:
        print(f"  Summary distribution plot saved to {os.path.join(OUTPUT_DIR, 'summary_delay_distribution.png')}")

print("Done.")
